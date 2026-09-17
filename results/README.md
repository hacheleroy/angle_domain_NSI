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
| `conventional_baselines/` | `src/conventional_baseline_comparison.py` |
| `conventional_timing/` | `src/benchmark_conventional.py` |

The conventional-baseline and conventional-timing outputs from the corrected
publication run are part of the immutable `reported/` snapshot. Their JSON
files record the exact CF, MV, and F-DMAS definitions, the pinned USTB
reference commit, fine F-DMAS axial sampling, exact lateral-batch validation,
GPU/software versions, and the publication-readiness checks. The reported
snapshot contains CSV/JSON evidence and manifests rather than the large cache
arrays or duplicate manuscript figures.

The GPU workflow was completed in two phases. The core-results archive was
`nsi_revision_results_2026-09-16.tar.gz` (SHA-256
`50034a3e46098fccff72b2ce3ce29ef222d5f5a939729ed5839aca60fbd66742`),
recorded by `revision_gpu_run_manifest_core_2026-09-16.json`. The corrected
conventional-comparison archive was
`nsi_conventional_results_2026-09-17-v2.tar.gz` (SHA-256
`78c52160f4b77d3ed945c2bc00a960e6268db6af35206c1c7c5e0629cca49916`),
recorded by
`revision_gpu_run_manifest_conventional_v2_2026-09-17.json`. Together these
manifests record completion of every requested workflow step, while
`revision_asset_manifest.json` records the final clean publication gate.
Absolute paths in these records document the author's run environment; they
are provenance, not required local paths for a rerun.
