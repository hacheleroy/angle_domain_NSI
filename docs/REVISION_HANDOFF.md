# PMB major-revision handoff

The repository contains the complete analysis program and immutable numerical
record for the Physics in Medicine & Biology revision. The full publication
run was completed on 17 September 2026 on the author's CUDA-capable NVIDIA
GPU. The corrected CSV/JSON outputs and run/asset manifests are frozen under
`results/reported/`; rerunning the workflow is optional reproducibility work,
not a remaining publication step.

## 1. Install the environment

Use the same CuPy, PyMUST, `mach-beamform`, NumPy, SciPy, Matplotlib, and h5py
environment as the original study. From the repository root,
install the test-only additions if needed:

```bash
python -m pip install -r requirements-test.txt
```

## 2. Fetch the public data

```bash
python scripts/fetch_datasets.py --dataset all
```

The acquisition script resumes interrupted downloads where supported, extracts
only the required PICMUS files, and validates every input against a recorded
SHA-256 digest. A successful run ends by writing `data/dataset_manifest.json`.

## 3. Run the revision analyses

```bash
python scripts/run_revision_gpu.py --device 0
```

The default workflow performs the analytic angular-null calculation,
point-target simulation, MBTrace analysis, PICMUS carotid analysis,
experimental PICMUS PSF study, transfer-aware timing suite, and manuscript
asset generation. It also runs the reviewer-facing conventional receive
CF-DAS/MV/F-DMAS comparison and its synchronized six-method timing benchmark.
It is resumable; completed steps and valid conventional-comparison case caches
are reused. Use `--force` only when an intentional full rerun is required.

On the WSL installation for which the Windows-provided CUDA driver library is
not selected automatically, keep the working override used during setup:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount()); print(cp.arange(3))"
```

If all earlier revision steps are already complete, the shorter equivalent is:

```bash
python scripts/run_revision_gpu.py --device 0 \
  --steps conventional-baselines,conventional-timing,manuscript-assets
```

The workflow validates cached timing settings before reuse and always rebuilds
the manuscript assets from the newest upstream results. The final asset step
is a strict publication gate. It fails rather than emit
tables when any of the following is true:

- the experimental PSF run was metadata-only or includes fewer than seven
  phantom targets;
- the approximately 37.5 mm spatial-variability group does not contain the
  left, central, and right targets;
- any point-target, MBTrace, carotid, or experimental-PSF `c` sweep omits one
  of 0.02, 0.05, 0.10, or 0.20;
- the timing run used quick settings, fewer than 10 warmups, or fewer than 50
  repetitions;
- a required primary or transfer-control benchmark case is missing, failed,
  or lacks one of the four comparison methods.
- the conventional comparison is metadata-only, quick, omits any of DAS,
  receive CF-DAS, MV, F-DMAS, Receive-NSI, or Angle-NSI, or does not contain
  the representative experimental PSF and both carotid views;
- the six-method post-delay benchmark used fewer than 10 warmups or 50
  repetitions, or is marked non-publication-ready.

## 4. Check the outputs

The run manifest is:

```text
results/generated/revision/revision_gpu_run_manifest.json
```

It must contain `"all_requested_steps_completed": true`. Publication-ready
assets are written under:

```text
results/generated/revision/manuscript_assets/
```

That directory must contain `revision_asset_manifest.json`,
`revision_results.tex`, the revision tables, and the standardized PNG/PDF
figures. The asset manifest records every source result and output file so that
the numerical values inserted into the manuscripts remain traceable.

The new reviewer-comparison assets are:

```text
table_conventional_baselines.tex
table_conventional_timing.tex
table_conventional_parameters.tex
reviewer_comparison/conventional_experimental_psf.png
reviewer_comparison/conventional_experimental_psf_profiles.png
reviewer_comparison/conventional_carotid_longitudinal.png
reviewer_comparison/conventional_carotid_cross_section.png
reviewer_comparison/conventional_timing.png
```

The primary metric for the nonlinear carotid comparison is gCNR. CR and CNR
are retained for completeness and are explicitly calculated on the linear
envelope; they should not replace gCNR in the main reviewer response.

## 5. Final manuscript checks

The generated assets have been synchronized to both Overleaf projects. Before
portal upload, retain the following final author-side checks:

1. compile the clean journal manuscript, highlighted manuscript,
   supplementary material, response letter, and arXiv manuscript;
2. confirm that no red `GPU RESULT PENDING` fallback marker is rendered;
3. confirm that every number in the response letter is supplied by
   `revision_results.tex` rather than transcribed manually;
4. inspect all tables and plots at final PDF scale;
5. recalculate the manuscript and abstract word counts after any final
   shortening;
6. run the journal revision-submission checklist before upload.

The GPU run and final asset sync are complete. Regenerate the submission PDFs
only if the manuscript text or metadata changes after this release.

## 6. IOP upload package

The decision letter gives a revision deadline of **26 October 2026**. The IOP
checklist maps the final files to the submission system as follows:

| File | Submission designation/check |
|---|---|
| Point-by-point response PDF | Upload in Step 1, “View and Respond to Decision Letter” |
| Highlighted complete manuscript PDF | “Complete Document for Review (PDF Only)”; figures and tables included |
| Clean `main.tex` | “Source Files”; no revision colour or tracked changes |
| Additional TeX components and high-resolution figures | “Source Files” in Step 3 |
| Clean manuscript PDF | “Source Files”; compiled from the clean source |
| Supplementary PDF and two supplementary movies | “Supplementary Data Files”; clean and titled/described |

Before upload, verify that the author list, affiliations, corresponding-author
email, funding/acknowledgements, and submission-form metadata agree exactly.
If the original submission used double-anonymous review, anonymise both the
response and highlighted PDF before uploading them.
