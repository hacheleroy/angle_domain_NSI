# Validation notes

The following checks were completed after adding the reviewer-requested
conventional baselines:

- Python syntax compilation passed for every source, workflow, and test
  module.
- Fifty-three CPU-side unit tests passed.
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

The earlier full GPU analyses were run by the author in the target CUDA
environment and their outputs are preserved in the supplied revision-results
archive. The conventional CF-DAS/MV/F-DMAS stages still require their first
full run on that machine. This workspace does not provide the CUDA hardware or
external datasets needed to execute them. Run the short conventional timing
`--quick` check in the target environment before launching the publication
configuration.
