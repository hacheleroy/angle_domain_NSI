# Accepted publication reference assets

This directory contains the 15 accepted figure images and checksums for all
25 figure, table and numerical-macro outputs in the final 15 October 2026
snapshot. Table/macro values are retained in the frozen CSV/JSON inputs;
their `.tex` fragments are regenerated and hash-checked, not tracked here.
Manuscript sources, bibliography,
journal class, compiled manuscript and reviewer correspondence are maintained
separately.

The bytes were recovered from commit
`44f90fc598969bdeaf78f858252477d3219b7cbc` and verified against its manifest.
`asset_manifest.json` schema 3 retains the original asset, numerical-input and
shared-input checksums, decoded-image hashes, and redraw/restoration lists.
It also declares the bounded Figure 4 display-rounding tolerance below.
The old manuscript-source checksums were removed because those sources are
outside this repository's current reproduction scope.

Run from the repository root in the locked CPython 3.12 Linux/WSL environment:

```bash
python scripts/reproduce_paper.py --check-only
python scripts/reproduce_paper.py
```

Eight of the fifteen figures are redrawn from saved data; seven are restored
from these reference images. All nine table files and the numerical macros
are regenerated. Figure 4 allows at most 32 pixels to shift by one equal RGB
level with unchanged opacity; CPU-dependent float32 logarithm rounding was
observed to change 12 pixels. All other figures require identical pixels.
Verified redraws use the reference PNG bytes to preserve accepted output bytes.
The output report distinguishes metadata-only and bounded raster differences.
The raw-data GPU workflow reconstructs new data and figures instead.

The reference files are immutable regression fixtures. New outputs belong in
`results/generated/`. A deliberate scientific revision should update the
underlying numerical record, references and manifest together, with a new
version and documented validation. Content is licensed under CC BY 4.0.
