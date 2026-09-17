# NSI validation protocol

These scripts reproduce the synchronized timing benchmark and robustness analyses reported in the manuscript and supplementary material. 

## Files

- `src/benchmark_nsi.py`: synchronized timing comparison
- `src/benchmark_scaling.py`: matrix/element/angle scaling and transfer controls
- `src/picmus_experimental_psf.py`: measured multi-position PICMUS PSFs
- `src/picmus_full_phantom_comparison.py`: full seven-target six-method figure
- `src/simulation_six_method_comparison.py`: six-method point-target figure
- `src/adaptive_beamforming.py`: shared receive CF, Capon MV, and DMAS
  primitives (NumPy/CuPy)
- `src/conventional_baseline_comparison.py`: representative six-method
  experimental PSF and carotid comparison
- `src/benchmark_conventional.py`: synchronized six-method post-delay timing
- `src/angular_null_theory.py`: paired-angle small-angle field model
- `src/simulation_robustness.py`: angular, perturbation, spatial-PSF and
  two-target tests
- `tests/test_validation_helpers.py`: CPU-only tests of the measurement helpers

Both analysis scripts save CSV and JSON records. The JSON files contain the
configuration needed to report the results reproducibly.

## 1. Installation check

These short executions check that imports, paths and output creation work. Do
not cite their values.

```bash
python -m unittest discover -s tests -v

python src/benchmark_nsi.py \
  --quick \
  --output-dir results/generated/timing_quick

python src/simulation_robustness.py \
  --quick \
  --output-dir results/generated/robustness_quick

python src/benchmark_conventional.py \
  --quick \
  --output-dir results/generated/conventional_timing_quick
```

The two GPU scripts require the same CuPy, PyMUST and mach-beamform environment
as `src/simulation_point_target.py`. The timing script does not require PyMUST.

## 2. Full fair timing benchmark

Recommended command:

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

This measures:

1. standard DAS/coherent compounding
2. a legacy angle-ensemble coherence control retained for numerical provenance
3. Receive-NSI from the two independent fields `U` and `Z_e`
4. Angle-NSI from `U` and `Z_theta`

Angle-NSI is reported both with streaming accumulation and with a
retained per-angle stack. The receive benchmark uses two simultaneous
beamformer outputs per angle; it is not the old three-independent-pass
implementation.

With the default `--receive-input-mode device-weighted`, every optimized method
transfers the same raw complex-IQ payload and the receive pair is formed on the
GPU inside the timed region. `--receive-input-mode host-stacked` reproduces the
older API condition in which Receive-NSI transfers twice the channel bytes.
Every summary row records exact logical input, geometry, output, and total
transfer bytes.

`--scope reconstruction` includes construction and GPU transfer of the scan
grid and transmit-arrival arrays in every timed execution. `--transfers both`
also includes the channel-data upload and final-image download. RF simulation,
RF-to-IQ demodulation, disk I/O and plotting remain outside the timed region and
are listed as exclusions in the JSON file.

To measure the kernel with preloaded inputs and geometry as a complementary
analysis:

```bash
python src/benchmark_nsi.py \
  --warmups 10 \
  --repetitions 50 \
  --scope kernel \
  --transfers none \
  --angle-storage-mode both \
  --output-dir results/generated/timing_kernel
```

By default, deterministic random complex IQ is used because signal content
does not change the work performed by the beamformer. To use a real IQ stack,
save `iq` with shape `(angles, samples, elements)` or
`(angles, elements, samples)` and, preferably, `angles_deg` in an NPZ file:

```bash
python src/benchmark_nsi.py \
  --iq-npz my_iq_stack.npz \
  --iq-key iq \
  --warmups 10 \
  --repetitions 50 \
  --scope reconstruction \
  --transfers both \
  --output-dir results/generated/timing_real_iq
```

The optional `--include-naive-reference` adds a clearly labelled three-field
receive implementation. It must not be used as the optimized comparator.

The generated files include:

- `nsi_timing_runs.csv`
- `nsi_timing_summary.csv`
- `nsi_timing_summary.json`
- `nsi_timing_summary.png`

The JSON records GPU/CPU models, CUDA and package versions, precision, grid,
element and angle counts, transfer scope and image-storage policy.

## 3. Reviewer-requested conventional beamformers

Validate the three PICMUS inputs without a GPU:

```bash
python src/conventional_baseline_comparison.py --metadata-only
```

Run the publication comparison on the GPU:

```bash
python src/conventional_baseline_comparison.py \
  --output-dir results/generated/conventional_baselines
```

The comparison uses the full 75-angle acquisition unless `--angle-count` is
explicitly supplied. Any reduced-angle or `--quick` run is marked
non-publication-ready. Completed PSF, longitudinal carotid, and cross-sectional
carotid reconstructions have independent signature-validated caches.

The implementation invariants are:

1. CF-DAS uses
   `|sum_m x_m|^2/(M_active*sum_m |x_m|^2)` per pixel and transmit angle;
2. MV uses analytic delayed channels, every overlapping contiguous subarray,
   `L=floor(M_active/2)`, loading `trace(R)/(100L)`, and a batched linear solve
   rather than an explicit inverse;
3. DMAS uses real delayed RF and
   `sign(s_i*s_j)*sqrt(abs(s_i*s_j))` for all `i<j`, followed by the fixed
   Kaiser band-pass around `2*f0` and an analytic-signal transform;
4. the optimized DMAS identity is tested numerically against the literal
   pair loop;
5. every method uses the same delay law, interpolation, active aperture, and
   transmit-angle set before its method-specific reduction.

The USTB reference files and exact reference commit are written into
`conventional_baseline_summary.json`; USTB is not a runtime dependency.

For the full-phantom overview, DAS, CF-DAS, Receive-NSI and Angle-NSI are
reconstructed on a 0.05 mm lateral grid. MV and DMAS use 0.15 and 0.10 mm
native lateral spacing, respectively, and are bilinearly resampled onto the
common display grid. The target-wise central-notch diagnostic uses separate
directly beamformed fine profiles and is therefore independent of this display
resampling.

The synchronized post-delay timing command is:

```bash
python src/benchmark_conventional.py \
  --warmups 10 \
  --repetitions 50 \
  --output-dir results/generated/conventional_timing
```

The timing boundary begins with complex-IQ and real-RF delayed channel tensors
already resident on the GPU. It includes each method-specific reduction, MV
covariance construction/solve, and DMAS FIR/Hilbert processing. It excludes
the identical delay calculation/interpolation, disk I/O, and transfers. Both
the boundary and leading-order operation counts are recorded in JSON.

## 4. Scaling and transfer-aware timing suite

```bash
python src/benchmark_scaling.py \
  --warmups 10 \
  --repetitions 50 \
  --output-dir results/generated/timing_scaling
```

The primary cases vary grids of 64x128, 128x256, and 256x512 pixels; 64, 128,
and 256 receive elements; and 9, 17, 33, and 75 transmit angles. Two controls
measure the baseline with all arrays preloaded and with host-stacked
Receive-NSI input. Cases are independently resumable. Median latency is
reported with IQR and a deterministic percentile-bootstrap 95% interval.

## 5. Experimental PSF and spatial-variability study

First fetch and checksum-verify the public inputs:

```bash
python scripts/fetch_datasets.py --dataset all
```

The CPU metadata check is:

```bash
python src/picmus_experimental_psf.py --metadata-only
```

The full GPU run is:

```bash
python src/picmus_experimental_psf.py \
  --output-dir results/generated/picmus_experimental_psf
```

Local maps use 0.02 mm sampling. Separate 0.002 mm lateral and 0.005 mm axial
profiles are beamformed directly from channel data at every target; no coarse
image interpolation is used. Its DAS, Receive-NSI and Angle-NSI profiles feed
the target-wise central-notch diagnostic in the full six-method phantom
figure. A notch is counted when a valley within 0.15 mm of the nominal target
has at least 6 dB prominence between local maxima. The output also includes
`c` sensitivity, the five-target depth series, and the three-target lateral
series near 37.5 mm.

## 6. Full robustness study

Recommended command:

```bash
python src/simulation_robustness.py \
  --output-dir results/generated/robustness
```

The default study contains:

- 5, 9, 13, 17 and 25 angles over the baseline 8-degree total span
- 4, 8, 12 and 16-degree total spans using 17 angles
- removal of the positive angle nearest +2 degrees
- the missing-angle case with raw sign weights and with a zero-mean,
  L1-normalized recentering
- complex-IQ SNR values of infinity, 40, 30, 20 and 10 dB
- inter-angle phase-jitter standard deviations of 0, 2, 5, 10 and 20 degrees
- 10 independent perturbation realizations per level
- single-target PSFs at depths 15, 20, 25 and 30 mm and lateral positions
  -4, 0 and +4 mm
- equal-amplitude two-target separations from 0.01 to 0.40 mm

The two-target criterion requires two peaks near their expected positions and
an inter-peak valley at least 6.02 dB below the weaker peak. This is deliberately
reported separately from the single-target nonlinear-output FWHM.

The full study can be long because it performs new RF simulations and uses a
0.390625 micrometre lateral grid. It prints progress after every scenario. The
individual `--skip-*` switches are useful for diagnosis, but a partial run is
marked non-publication-ready in `robustness_summary.json`.

The generated output directory includes:

- `robustness_summary.json`
- `robustness_angle_sweeps.csv` and `.png`
- `robustness_noise_phase_sweeps.csv` and `.png`
- `robustness_spatial_psf.csv` and `.png`
- `robustness_two_target_resolvability.csv` and `.png`
- `robustness_two_target_profiles.csv`

## Interpretation safeguards

- Widths are connected main-lobe envelope widths with linearly interpolated
  threshold crossings.
- Each method is measured through its own baseline axial PSF peak.
- Perturbation sweeps use that fixed baseline depth so that noise does not
  select a different axial slice for each realization.
- The missing-angle raw-sign case intentionally has a nonzero weight sum. The
  recentered case tests a simple mitigation and is labelled separately.
- A very narrow single-target NSI FWHM is not claimed to be physical target
  resolvability; the explicit two-target analysis is the relevant evidence.
- A DAS-to-NSI peak match is positional concordance against a reconstruction,
  not an estimate of biological sensitivity. The MBTrace audit separates
  within-tolerance matches, displacements within a wider tracking radius, and
  cases with no nearby detected peak; the last category cannot distinguish
  threshold suppression from true absence.
