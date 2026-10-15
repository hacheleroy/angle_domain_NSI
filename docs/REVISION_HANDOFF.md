# Final paper reproduction handoff

The manuscript content and figure/table set are frozen on 15 October 2026.
No additional GPU run is required to regenerate the published artifacts.

```bash
python -m pip install -r requirements-cpu.txt
python scripts/reproduce_paper.py
```

Outputs are under `results/generated/paper_assets/`. The command verifies all
25 used artifacts against `paper/asset_manifest.json`; no unused figure or
table is emitted. The clean scientific LaTeX sources are in `paper/`.
The journal response and highlighted review document remain in the journal
Overleaf project and resubmission package.

For a complete new analysis from public RF data:

```bash
python scripts/fetch_datasets.py --dataset all
python scripts/reproduce_paper.py --from-raw --device 0
```

The full workflow includes the seven-target means/sample SD and the fixed-ROI
inclusion gCNR that were added after the earlier release. The WSL checkpoint
strategy remains in use. Consult `README.md`, `data/README.md` and
`docs/REPRODUCIBILITY.md` for environments, data placement and provenance.
