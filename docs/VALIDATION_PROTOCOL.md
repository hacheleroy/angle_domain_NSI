# NSI validation protocol

These scripts implement mandatory items 4 and 5 of the PMB revision report.
They do not edit the manuscript. Run them on the same workstation and software
environment used for the revised reconstructions, then return the complete
output directories for integration into the Note and supplementary material.

## Files

- `src/benchmark_nsi.py`: synchronized timing comparison
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
```

The two GPU scripts require the same CuPy, PyMUST and mach-beamform environment
as `src/simulation_point_target.py`. The timing script does not require PyMUST.

## 2. Full fair timing benchmark (mandatory item 4)

Recommended command:

```bash
python src/benchmark_nsi.py \
  --warmups 10 \
  --repetitions 50 \
  --scope reconstruction \
  --transfers both \
  --angle-storage-mode both \
  --output-dir results/generated/timing
```

This measures:

1. standard DAS/coherent compounding
2. receive-domain NSI from the two independent fields `U` and `Z_e`
3. angle-domain NSI from `U` and `Z_theta`

The angle-domain method is reported both with streaming accumulation and with a
retained per-angle stack. The receive benchmark uses two simultaneous
beamformer outputs per angle; it is not the old three-independent-pass
implementation.

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

Return these files:

- `nsi_timing_runs.csv`
- `nsi_timing_summary.csv`
- `nsi_timing_summary.json`
- `nsi_timing_summary.png`

The JSON records GPU/CPU models, CUDA and package versions, precision, grid,
element and angle counts, transfer scope and image-storage policy.

## 3. Full robustness study (mandatory item 5)

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

Return the whole output directory, particularly:

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
