# Validation notes

The following checks were completed during repository packaging:

- Python syntax compilation passed for all six source modules and all three
  test modules.
- Fifteen CPU-side unit tests passed.
- The tests cover symmetric angular weights and broadside zero weighting,
  timing statistics, the DAS/receive/angle benchmark execution paths with a
  NumPy stand-in backend, equivalence of streaming and stored angle-domain
  accumulation, equivalence of two-field and naive receive NSI, interpolated
  half-amplitude width measurement, missing-angle weight recentering, the
  two-target dip criterion, and generation of all five output figure types.
- Both command-line interfaces and help text execute before loading GPU-only
  packages.

The full GPU analyses were run by the author in the target CUDA environment;
their CSV and JSON outputs are preserved under `results/reported/`. The
packaging workspace does not provide the CUDA hardware or external datasets
needed to repeat those full reconstructions. Run the short `--quick` checks in
the target environment before launching the publication configurations.
