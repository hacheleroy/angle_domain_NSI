# Result records

`reference/` contains the 25 accepted figure/table/macro assets for the final
15 October 2026 snapshot. `reference/asset_manifest.json` records their
checksums, the numerical inputs and the shared robustness records.
`reported/paper_inputs/` contains the frozen maps, profiles, scalar summaries
and timing repetitions used for this snapshot.

From the repository root, in the locked CPython 3.12 Linux environment:

```bash
python scripts/reproduce_paper.py --check-only
python scripts/reproduce_paper.py
```

Outputs go to `generated/paper_assets/`. Eight figures are redrawn and seven
restored from references; nine tables and one numerical-macro file are
regenerated. Pixel-identical PNG redraws retain reference metadata bytes.
The generated report identifies these operations. Manuscript sources are
maintained separately and are unnecessary for replay.

Other `reported/` subdirectories preserve earlier September runs and their
manifests. Their records are historical, except shared robustness files
explicitly selected and checksummed by the reference manifest. Legacy method
labels are translated through `src/method_names.py`.

`exploratory/` holds analyses outside the manuscript claims. `generated/` is
ignored by Git and receives new CPU examples, reanalyses, RF reconstructions
and asset outputs. Frozen references and reported records must not be silently
overwritten. External RF data are acquired separately using
`scripts/fetch_datasets.py` and remain subject to their source terms.
