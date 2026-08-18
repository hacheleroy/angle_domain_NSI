# Result files

`reported/` contains the machine-readable CSV and JSON outputs used for the
manuscript and supplementary material. They are committed as an immutable
record of the reported run.

`exploratory/` contains analyses retained for transparency but not used for a
manuscript claim. In particular, the two-target output was excluded from the
Note because the current nonlinear peak-valley metric is not an adequate
standalone resolution criterion.

New executions write to `generated/` by default. That directory is ignored by
Git so rerunning a script does not silently replace the reported record.

The B-mode CSV/JSON files preserve the exact metadata emitted by the original
v4 run, including its absolute source paths and the legacy display labels
`Receive NSI` and `Angle NSI`. In the cleaned source and manuscript these map
to `Conventional NSI` and `Angular NSI`, respectively. The recorded paths are
provenance only; new runs use the portable paths documented in `data/README.md`.

Reported subdirectories map to scripts as follows:

| Directory | Generating script |
|---|---|
| `point_target/` | `src/simulation_point_target.py` |
| `robustness/` | `src/simulation_robustness.py` |
| `timing/` | `src/benchmark_nsi.py` |
| `bmode/` | `src/bmode_picmus.py` |
| `doppler/` | `src/doppler_mbtrace.py` |
