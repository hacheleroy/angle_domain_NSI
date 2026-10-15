# Result records

The final 15 October 2026 paper uses the frozen numerical inputs in
`reported/paper_inputs/` and the used figures/tables in `../paper/`.
`paper/asset_manifest.json` is the authoritative source/checksum map.

Run `python scripts/reproduce_paper.py` to regenerate and verify the complete
25-asset paper set without CUDA. New outputs go to `generated/paper_assets/`.
The replay report distinguishes redrawn plots from archived rendered panels.

The other `reported/` subdirectories and manifests preserve the earlier
September release and revision runs. They are historical records and are not
the final paper's complete result set. The robustness CSVs remain shared with
the final paper and are explicitly selected by the asset materializer.
Legacy method labels in original records are translated through
`src/method_names.py`; the six display names are DAS, CF-DAS, MV, DMAS,
Receive-NSI and Angle-NSI.

`exploratory/` contains analyses outside the manuscript claims.
`generated/` is ignored by Git so new calculations cannot silently overwrite
the frozen publication inputs. External RF data remain outside the result
snapshot and are acquired using `scripts/fetch_datasets.py`.
