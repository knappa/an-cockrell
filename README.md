# An-Cockrell Reimplementation

A Python reimplementation of the NetLogo agent-based model from:

> An, G. & Cockrell, C. (2021). "Comparative Computational Modeling of the Bat and Human Immune Response to Viral Infection with the Comparative Biology Immune Agent Based Model." *Viruses*, 13(8), 1620. https://doi.org/10.3390/v13081620

The original NetLogo model and its documentation are in [`Comparative-Biology-Immune-ABM/`](Comparative-Biology-Immune-ABM/). A transcription of the original interface documentation is in [`src/an_cockrell/ORIG_DOCS.md`](src/an_cockrell/ORIG_DOCS.md).

## What the model does

The model simulates the innate immune response to a respiratory viral infection on a 51×51 spatial grid. 

**Agents:** epithelial cells, macrophages, NK cells, dendritic cells (DCs), PMNs, and activated endothelial cells.

**Molecular fields:** extracellular virus, T1IFN, IFN-γ, TNF, IL-1, IL-6, IL-8, IL-10, IL-12, IL-18, P/DAMPs, PAF, ROS.

Each time step represents one hour. A full run is 2016 steps (14 days).

## Installation

```bash
pip install -e .
```

Requires Python &geq; 3.8. Dependencies (installed automatically): `attrs`, `numpy`, `h5py`, `matplotlib`, `scikit-learn`.

## Quick start

```python
import an_cockrell

model = an_cockrell.AnCockrellModel(
    GRID_WIDTH=51,
    GRID_HEIGHT=51,
    is_bat=False,
    init_inoculum=100,
    init_dcs=50,
    init_nks=25,
    init_macros=50,
)

# advance one hour
model.time_step()

# run for 14 days
for _ in range(2016):
    model.time_step()

print(f"System health: {model.system_health:.1f}%")
print(f"Infected cells: {model.infected_epithelium_count}")
```

See [PARAMETERS.md](PARAMETERS.md) for the complete parameter reference.

## Visualization

```python
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
model.plot_agents(axs[0])                              # cells and immune agents
model.plot_field(axs[1], field_name="extracellular_virus")  # any molecular field
for ax in axs:
    ax.set_aspect(1)
plt.show()
```

**Agent color key:**
- Blue squares — healthy epithelial cells
- Yellow squares — infected epithelial cells
- Grey squares — necrotic epithelial cells
- Grey pentagons — apoptosed epithelial cells
- Green circles — macrophages (larger = at phagocytosis limit)
- Orange circles — NK cells
- Light blue triangles — dendritic cells
- Pink square outlines — activated endothelial cells
- Small white circles — PMNs

**Available fields for `plot_field`:** `extracellular_virus`, `T1IFN`, `IFNg`, `TNF`, `IL1`, `IL6`, `IL8`, `IL10`, `IL12`, `IL18`, `P_DAMPS`, `PAF`, `ROS`.

## Saving and loading state

```python
# save state at every step to an HDF5 file
model.save("run.hdf5")

# load a saved state
model = an_cockrell.AnCockrellModel.load("run.hdf5", time=500)
```

## Analysis tools

The `analysis/` directory contains:

- **`An_Cockrell_model.ipynb`** — Jupyter notebook with example runs, animations, and time-series plots.
- **`an-cockrell-runner.py`** — Latin hypercube sampling for sensitivity analysis.
- **`an-cockrell-viewer.py`** — Viewer for saved HDF5 runs.
- **`slurm-job.sh`** — SLURM job script for running the sensitivity analysis on HiPerGator.
