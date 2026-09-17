# Validation notes

The following checks were completed after adding the reviewer-requested
conventional baselines:

- Python syntax compilation passed for every source, workflow, and test
  module.
- Fifty-seven CPU-side unit tests passed.
- The tests cover symmetric angular weights and broadside zero weighting,
  timing statistics, the DAS/receive/angle benchmark execution paths with a
  NumPy stand-in backend, equivalence of streaming and stored angle-domain
  accumulation, equivalence of two-field and naive receive NSI, interpolated
  half-amplitude width measurement, missing-angle weight recentering, the
  two-target dip criterion, and generation of all five original output figure
  types.
- New tests cover the receive-CF definition and bounds, equivalence of the
  optimized and literal F-DMAS pair sums, F-DMAS sampling/filter guards,
  Hilbert formation, MV focused-signal preservation, dynamic contiguous
  apertures, temporal averaging, six-method NumPy execution, bilinear F-DMAS
  resampling, gCNR limiting cases, and strict cache/publication gates.
- The relevant command-line interfaces and help text execute before loading
  GPU-only packages.
- A synthetic HDF5 metadata-only run validated all four input paths and the
  serialized fixed-parameter/provenance record for the new comparison.

The publication workflow was completed by the author in two phases on 16--17
September 2026 with an NVIDIA RTX A2000, CuPy 14.1.1 and CUDA runtime 12.9.
The corrected conventional-comparison v2 archive passed the strict
publication gate: all six methods were present, the experimental PSF and both
carotid views were complete, every F-DMAS lateral line contained finite
positive signal, and the post-delay timing used 10 warm-ups and 50
repetitions. The selected machine-readable outputs and the two run manifests,
covering all requested workflow steps, are frozen under `results/reported/`
alongside the final asset manifest.

The final F-DMAS implementation releases the preceding IQ-method GPU buffers,
processes independent lateral batches, validates every output line, and
recursively bisects a failed batch down to a single-line fallback. Tests verify
that chunking and fallback preserve the unchunked beamformer output. The
publication archive exercised this fallback for the longitudinal carotid case
and subsequently passed complete-image validation.
