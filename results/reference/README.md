# Accepted publication reference assets

This directory contains only the 25 figure, table and numerical-macro assets
used in the final 15 October 2026 snapshot. Manuscript sources, bibliography,
journal class, compiled manuscript and reviewer correspondence are maintained
separately.

The bytes were recovered from commit
`44f90fc598969bdeaf78f858252477d3219b7cbc` and verified against its manifest.
`asset_manifest.json` schema 2 retains the original asset, numerical-input and
shared-input checksums, decoded-image hashes, and redraw/restoration lists.
The old manuscript-source checksums were removed because those sources are
outside this repository's current reproduction scope.

Run from the repository root in the locked CPython 3.12 Linux/WSL environment:

```bash
python scripts/reproduce_paper.py --check-only
python scripts/reproduce_paper.py
```

Eight of the fifteen figures are redrawn from saved data; seven are restored
from these reference images. All nine table files and the numerical macros
are regenerated. Pixel-equal redraws use the reference PNG bytes to preserve
original metadata. The output report records these operations explicitly.
The raw-data GPU workflow reconstructs new data and figures instead.

The reference files are immutable regression fixtures. New outputs belong in
`results/generated/`. A deliberate scientific revision should update the
underlying numerical record, references and manifest together, with a new
version and documented validation. Content is licensed under CC BY 4.0.
