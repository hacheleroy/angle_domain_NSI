# Optional CUDA reconstruction environment

The core `angle_nsi` API and CPU demo work without CUDA. This environment is
for regenerating RF-based experiments, simulations and benchmarks.

Use a dedicated CPython 3.12 Linux/WSL environment and an NVIDIA GPU/driver
compatible with the CuPy CUDA 12.x wheels. The historical study used an RTX
A2000 with CuPy 14.1.1 and CUDA runtime 12.9; per-run JSON records retain the
versions actually captured, including historical Python/library versions.

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements.txt
python -m pip install . --no-deps
python -m pip check
```

`requirements.txt` freezes every directly used reconstruction dependency,
including PyMUST and threadpoolctl. The `[gpu]` package extra declares the same
direct requirements. These additional maintenance pins are prospective; they
are not retroactively presented as the full environment of historical runs.
`requirements-gpu-lock.txt` freezes the resolved transitive GPU environment
through a constraints directive in `requirements.txt`. Resolution was checked
on CPython 3.12/Linux without installing or executing CUDA. The separate
validated CPU replay lock is `requirements-cpu.txt`.

## Device and small reconstruction check

```bash
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceProperties(0)); print(cp.arange(4).sum().get())"
python src/benchmark_nsi.py --quick --output-dir results/generated/timing_smoke
```

`--quick` checks execution and output creation; its timings are not publication
measurements. Confirm device execution before downloading the larger datasets.
On WSL setups needing Windows CUDA libraries first in the search path:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

## Data and full run

Follow [data/README.md](../data/README.md). Box may require downloading MBTrace
manually. PICMUS files are checked against the recorded SHA-256 values.

```bash
python scripts/fetch_datasets.py --dataset all
python scripts/reproduce_paper.py --from-raw --device 0
```

The workflow saves run manifests, stdout/stderr and package versions under
`results/generated/revision/`. It resumes validated outputs and uses isolated
lateral tile checkpoints for large WSL stages. The seven-target stage uses
NumPy/SciPy on two CPU workers by default, checking against saved GPU profiles
before processing all targets. Adjust `--psf-workers` through the direct runner
when needed. Use its `--steps` option for individual stages and `--force` for
an intentional rerun.

VRAM demand and elapsed time depend on grid, channels, angles and method;
MV is substantially more demanding than the Angle-NSI stack operation. Tile
and chunk controls are exposed by the full-phantom/comparison scripts. Consult
each `--help` and retain the run manifest when comparing hardware. No universal
minimum VRAM or run duration is claimed without a measured configuration.

CPU CI does not execute this complete CUDA path. A successful dependency
resolution or smoke test alone does not establish agreement with historical
reconstructions. Record a complete hardware run separately when validating
this optional environment.
