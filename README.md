# \# Rare-spin detection via driven nuclear-spin bath dynamics (numerical study)

# 

# This repository contains the Python code used to generate the simulation data and reproduce the figures/summary plots used in the accompanying dissertation.

# 

# \## Repository contents

# 

# Core simulation + sweep:

# \- `dipolar\_ensemble\_with\_rare.py` — builds the heteronuclear dipolar spin ensemble Hamiltonian and runs a single time-evolution (QuTiP).

# \- `sweep\_sea\_detuning.py` — runs a \*sea detuning sweep\* (rare drive OFF vs ON) for one chosen sea drive amplitude and produces a per-sweep PDF plus per-detuning outputs.

# 

# Post-processing / report generation:

# \- `reprocess\_sweep\_results.py` — reprocesses existing sweep output folders (e.g., if you change coarse-graining window size or fitting choices) and regenerates summary artifacts.

# \- `2D\_sweep\_report.py` — aggregates \*\*multiple\*\* sweep folders (different sea drive amplitudes) into a single PDF summary (contrast vs η, contrast vs scaled detuning, etc.).

# \- `2D\_sweep\_report\_stable\_region.py` — same as above but also computes a “stable” scaled-detuning band using a pass-fraction criterion and emits an additional plot + JSON stats.

# 

# ---

# 

# \## Quick start

# 

# \### 1) Create a Python environment

# 

# Recommended: Python 3.10+.

# 

# ```bash

# python -m venv .venv

# \# Windows PowerShell:

# .\\.venv\\Scripts\\Activate.ps1

# \# macOS/Linux:

# \# source .venv/bin/activate

# ```

# 

# Install dependencies:

# 

# ```bash

# pip install --upgrade pip

# pip install numpy matplotlib qutip

# ```

# 

# Notes:

# \- `qutip` will pull in its scientific stack dependencies automatically.

# \- Some scripts use `tkinter` for folder selection; on Windows it is typically included with Python. On some Linux distros you may need to install it separately.

# 

# ---

# 

# \## Generating data (running the simulations)

# 

# \### A) Run a single detuning sweep

# 

# `sweep\_sea\_detuning.py` is the entry point for generating a sweep folder containing raw time traces, extracted metrics, and a per-sweep PDF report.

# 

# 1\. Open `sweep\_sea\_detuning.py` and edit the configuration in the `if \_\_name\_\_ == "\_\_main\_\_":` section:

# &nbsp;  - `global\_params`: geometry, coupling scale, evolution time, solver options, etc.

# &nbsp;  - `sweep`: sea drive parameters (e.g., `f1A\_Hz`) and the detuning list for that drive.

# 2\. Run:

# 

# ```bash

# python sweep\_sea\_detuning.py

# ```

# 

# \### Outputs (per sweep)

# 

# A new results folder is created (by default under `results/`), containing:

# \- `sea\_detuning\_report.pdf` — PDF report for that sweep.

# \- `summary.json` — machine-readable per-sweep summary (global params + all detuning-point results).

# \- `sweep\_results.csv` — tabular export of the per-detuning-point metrics.

# \- A set of `detuning\_XXXXXHz/` subfolders, one per detuning, each containing:

# &nbsp; - raw time traces (`time\_and\_obs\_\*.npz`)

# &nbsp; - a per-detuning PDF report (`detuning\_...\_report.pdf`)

# &nbsp; - `params.json` (parameters for that detuning point)

# &nbsp; - `freqs.json` (derived frequencies such as effective fields)

# &nbsp; - `metrics.json` (extracted slopes/contrast/etc.)

# 

# ---

# 

# \## Reprocessing existing data (without re-running the solver)

# 

# If you already ran sweeps and want to regenerate plots/metrics (e.g., change the coarse-graining window length, or update the linear-fit windowing), use:

# 

# ```bash

# python reprocess\_sweep\_results.py

# ```

# 

# This script will prompt you to select the root results directory and will then:

# \- iterate over sweep folders that contain `summary.json`,

# \- re-load the saved `.npz` traces,

# \- recompute derived quantities,

# \- write `summary\_reprocessed.json`,

# \- regenerate a PDF report (`sea\_detuning\_report\_reprocessed.pdf`).

# 

# ---

# 

# \## Reproducing dissertation summary figures (aggregating multiple sweeps)

# 

# \### A) Create the 2D summary report (all sweeps)

# 

# After you have multiple sweep folders (e.g., different `f1A` values), run:

# 

# ```bash

# python 2D\_sweep\_report.py --root path/to/results --output contrast\_vs\_coupling\_summary.pdf

# ```

# 

# Where `--root` is the directory that contains many sweep folders (each with a `summary.json`).

# 

# This produces a single PDF and also writes out the key figures as PNGs in the working directory:

# \- `01\_contrast\_vs\_eta.png`

# \- `02\_contrast\_vs\_scaled\_detuning.png`

# 

# \### B) Add the “stable scaled-detuning band” analysis

# 

# To reproduce the stable-region plots (pass fraction vs scaled detuning) and export the chosen band to JSON:

# 

# ```bash

# python 2D\_sweep\_report\_stable\_region.py \\

# &nbsp; --root path/to/results \\

# &nbsp; --output contrast\_vs\_coupling\_summary.pdf \\

# &nbsp; --C0 0.15 \\

# &nbsp; --min-p 0.6 \\

# &nbsp; --min-count 5 \\

# &nbsp; --bin-decimals 2 \\

# &nbsp; --stable-json stable\_region\_stats.json \\

# &nbsp; --add-stability-page

# ```

# 

# This will additionally emit:

# \- `05\_pass\_fraction\_vs\_scaled\_detuning.png`

# \- `stable\_region\_stats.json`

# 

# ---

# 

# \## Mapping scripts to figures used in the dissertation

# 

# Typical figure provenance:

# 

# \- Representative time traces + coarse-grained envelopes (single parameter point):

# &nbsp; - produced in the per-detuning reports generated by `sweep\_sea\_detuning.py`

# \- `total\_contrast\_vs\_eta` / “Contrast vs η”:

# &nbsp; - produced by `2D\_sweep\_report.py` (or the stable-region variant)

# \- `total\_contrast\_vs\_scaled\_detuning` / “Contrast vs δA / f1A”:

# &nbsp; - produced by `2D\_sweep\_report.py` (or the stable-region variant)

# \- Pass fraction vs scaled detuning (stable region selection):

# &nbsp; - produced by `2D\_sweep\_report\_stable\_region.py`

# 

# ---

# 

# \## Reproducibility notes

# 

# \- The simulations are deterministic for a fixed Hamiltonian and solver settings.

# \- If you change any of the following, you must re-run the sweeps (not just reprocess):

# &nbsp; - geometry/positions, coupling scale, species gyromagnetic ratios,

# &nbsp; - Hamiltonian terms, initial state preparation,

# &nbsp; - time grid and total simulation time (because the raw traces change).

# \- If you only change the \*\*analysis choices\*\* (coarse-graining window, fit window, thresholds), you can reprocess.

