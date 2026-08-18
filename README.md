# Angle-domain null subtraction imaging

Reproducibility code and reported numerical outputs for the manuscript
"Angle-domain null subtraction imaging from beamformed plane-wave data" by
Henri Leroy.

The repository compares three reconstructions:

1. delay-and-sum (DAS) with coherent plane-wave compounding;
2. conventional receive-domain null subtraction imaging (NSI); and
3. angular NSI, in which the zero-sum weights are applied across the
   per-angle complex image stack.

Angular NSI is a post-receive-beamforming construction. It is not claimed to
be algebraically identical to conventional receive-domain NSI.

## Repository layout

```text
src/                 Analysis and reconstruction scripts
tests/               CPU-only unit tests for algebra and measurement helpers
data/                Placement instructions for external data (not included)
results/reported/    CSV/JSON outputs used in the manuscript
results/exploratory/ Additional analyses not used for manuscript claims
docs/                Validation protocol and implementation notes
```

The stable script names replace the version suffixes used during manuscript
development. No beamforming or measurement formula was changed during this
packaging step.

## Software environment

The synchronized benchmark reported in the manuscript used Python 3.14.4,
NumPy 2.2.6, CuPy 14.1.1, Matplotlib 3.10.9 and MACH-beamform 0.1.3 on an
NVIDIA RTX A2000. Install the dependencies in a fresh environment with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` selects the CUDA 12 CuPy wheel used for the reported
benchmark. On another CUDA major version, install the corresponding CuPy
package instead. The CPU-only tests require only NumPy, SciPy and Matplotlib
and can instead be installed with `requirements-test.txt`.

## Verify the installation

From the repository root:

```bash
python -m unittest discover -s tests -v
python src/benchmark_nsi.py --help
python src/simulation_robustness.py --help
```

The full GPU scripts require a CUDA-capable NVIDIA GPU. Quick GPU checks are:

```bash
python src/benchmark_nsi.py --quick
python src/simulation_robustness.py --quick
```

Quick-mode values are engineering checks and must not be used in publications.

## Reproduce the reported analyses

### Point-target simulation

```bash
python src/simulation_point_target.py
```

Outputs are written to `results/generated/point_target`. Override this with
the `NSI_OUTPUT_DIR` environment variable.

### Robustness sweeps

```bash
python src/simulation_robustness.py \
  --output-dir results/generated/robustness
```

The angle-count, angular-span, missing-angle, noise, phase-jitter and spatial
PSF outputs are reported in the manuscript or supplement. The two-target
analysis is retained as exploratory output and is not used for a manuscript
claim.

### Synchronized timing benchmark

```bash
python src/benchmark_nsi.py \
  --warmups 10 \
  --repetitions 50 \
  --scope reconstruction \
  --transfers both \
  --angle-storage-mode both \
  --output-dir results/generated/timing
```

The benchmark validates streaming and stored-stack angular NSI using
scale-aware numerical tolerances because complex64 summation order can cause
small round-off differences. See `docs/VALIDATION_PROTOCOL.md` for the timing
scope and optional commands.

### PICMUS carotid B-mode data

Place the two PICMUS in-vivo datasets as described in `data/README.md`, then:

```bash
PICMUS_DATA_DIR=/path/to/PICMUS/in_vivo \
NSI_OUTPUT_DIR=results/generated/bmode \
python src/bmode_picmus.py
```

### Open-NSI MBTrace Doppler data

Place `MBTrace.mat` as described in `data/README.md`, then:

```bash
OPEN_NSI_MBTRACE_FILE=/path/to/MBTrace.mat \
NSI_OUTPUT_DIR=results/generated/doppler \
python src/doppler_mbtrace.py
```

The first run creates a neighboring `.npy` cache. The Doppler pipeline imports
the matched trace-width measurements from `src/trace_width_analysis.py`.

## Interpretation and provenance

- Reported widths are connected main-lobe widths at -6 dB in amplitude.
- The very narrow conventional-NSI value is explicitly described as an
  apparent nonlinear-output width, not a conventional linear-system PSF.
- Conventional NSI is implemented from two independent complex fields,
  `U` and `Z_e`; its two DC-offset fields are formed algebraically.
- Angular streaming and stored-stack reductions are numerically equivalent
  within the recorded complex64 tolerances.
- External datasets are not redistributed. Their placement and provenance are
  documented under `data/`.

The committed CSV and JSON files under `results/reported/` are the values used
in the manuscript. New runs write to `results/generated/`, which is ignored by
Git so reported results cannot be overwritten accidentally.

## Citation, DOI and license

License file is provided in 'LICENSE'. Citation metadata are provided in `CITATION.cff`. 
If you find this code or method useful in your research, please cite the paper:

```bibtex
@article{leroy2026angledomain,
  title={Angle-domain null subtraction imaging from beamformed plane-wave data},
  author={Henri Leroy},
  journal={arXiv preprint arXiv:2608.XXXXX},
  year={2026}
}
