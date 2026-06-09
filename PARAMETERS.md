# Parameter Reference

All parameters are keyword-only constructor arguments to `AnCockrellModel`.
Parameters with no listed default are required.

---

## Init-only parameters

These parameters affect only the initial state of the model and have no further effect on dynamics.

| Parameter                               | Type  | Default | Description                                                         |
|-----------------------------------------|-------|---------|---------------------------------------------------------------------|
| `init_inoculum`                         | `int` | (req.)  | Number of grid sites seeded with extracellular virus at t=0         |
| `init_dcs`                              | `int` | (req.)  | Initial dendritic cell count                                        |
| `init_nks`                              | `int` | (req.)  | Initial NK cell count                                               |
| `init_macros`                           | `int` | (req.)  | Initial macrophage count                                            |
| `extracellular_virus_init_amount_lower` | `int` | 80      | Minimum virus placed at each inoculation site                       |
| `extracellular_virus_init_amount_range` | `int` | 40      | Range added to lower (amount ~ Uniform[lower, lower+range))         |
| `epi_cell_membrane_init_lower`          | `int` | 975     | Minimum initial cell-membrane value (consumed by viral exocytosis)  |
| `epi_cell_membrane_init_range`          | `int` | 51      | Range of initial cell-membrane values                               |

## Macrophage phagocytosis

| Parameter                | Type    | Default | Description                                                                  |
|--------------------------|---------|---------|------------------------------------------------------------------------------|
| `macro_phago_recovery`   | `float` | 0.5     | Per-step decrement of the phagocytosis counter (models internal processing)  |
| `macro_phago_limit`      | `int`   | 1000    | Counter threshold above which a macrophage is "full" and stops phagocytosing |
| `macro_max_virus_uptake` | `float` | 10.0    | Maximum extracellular virus consumed per macrophage per step                 |

## Macrophage activation and cytokine secretion

| Parameter                            | Type    | Default | Description                                              |
|--------------------------------------|---------|---------|----------------------------------------------------------|
| `macro_activation_threshold`         | `float` | 5.0     | Activation level threshold for pro-inflammatory response |
| `macro_antiactivation_threshold`     | `float` | 5.0     | Threshold below which anti-inflammatory response fires   |
| `activated_macro_il8_secretion`      | `float` | 1.0     | IL-8 secreted per activated macrophage per step          |
| `activated_macro_il12_secretion`     | `float` | 0.5     | IL-12 secreted per activated macrophage per step         |
| `activated_macro_tnf_secretion`      | `float` | 1.0     | TNF secreted per activated macrophage per step           |
| `activated_macro_il6_secretion`      | `float` | 0.4     | IL-6 secreted per activated macrophage per step          |
| `activated_macro_il10_secretion`     | `float` | 1.0     | IL-10 secreted per activated macrophage per step         |
| `antiactivated_macro_il10_secretion` | `float` | 0.5     | IL-10 secreted per anti-activated macrophage per step    |

## Inflammasome and pyroptosis

| Parameter                               | Type    | Default | Description                                                     |
|-----------------------------------------|---------|---------|-----------------------------------------------------------------|
| `inflammasome_priming_threshold`        | `float` | 1.0     | P/DAMP + TNF level required to prime the inflammasome           |
| `inflammasome_activation_threshold`     | `int`   | 10      | virus_eaten / 10 required to activate a primed inflammasome     |
| `inflammasome_il1_secretion`            | `float` | 1.0     | IL-1 secreted per active-inflammasome macrophage per step       |
| `inflammasome_macro_pre_il1_secretion`  | `float` | 5.0     | Pre-IL-1 accumulated per step (released in burst at pyroptosis) |
| `inflammasome_il18_secretion`           | `float` | 1.0     | IL-18 secreted per active-inflammasome macrophage per step      |
| `inflammasome_macro_pre_il18_secretion` | `float` | 0.5     | Pre-IL-18 accumulated per step                                  |
| `pyroptosis_macro_pdamps_secretion`     | `float` | 10.0    | P/DAMPs released when a macrophage undergoes pyroptosis         |

## Epithelial cells

| Parameter                                          | Type    | Default | Description                                                            |
|----------------------------------------------------|---------|---------|------------------------------------------------------------------------|
| `susceptibility_to_infection`                      | `int`   | 77      | Base probability (out of 100) of a healthy cell becoming infected      |
| `viral_incubation_threshold`                       | `int`   | 60      | Intracellular virus count at which a cell begins releasing virus       |
| `epi_apoptosis_threshold_lower`                    | `int`   | 450     | Minimum apoptosis counter threshold at initialisation                  |
| `epi_apoptosis_threshold_range`                    | `int`   | 100     | Range added to lower bound (threshold ~ Uniform[lower, lower+range))   |
| `epi_apoptosis_threshold_lower_regrow`             | `int`   | 475     | Minimum threshold for regrown cells                                    |
| `epi_apoptosis_threshold_range_regrow`             | `int`   | 51      | Range for regrown cells                                                |
| `epi_regrowth_counter_threshold`                   | `int`   | 432     | Steps an empty patch must have >=3 healthy neighbours before regrowing |
| `infected_epithelium_ros_damage_counter_threshold` | `int`   | 10      | ROS-damage counter at which an infected cell dies by necrosis          |
| `epithelium_ros_damage_counter_threshold`          | `int`   | 2       | ROS-damage counter at which a healthy cell dies by necrosis            |
| `epithelium_pdamps_secretion_on_death`             | `float` | 10.0    | P/DAMPs released when a healthy cell dies                              |
| `dead_epithelium_pdamps_burst_secretion`           | `float` | 10.0    | P/DAMPs released when an infected cell membrane collapses              |
| `dead_epithelium_pdamps_secretion`                 | `float` | 1.0     | P/DAMPs released per dead cell per step                                |
| `epi_max_tnf_uptake`                               | `float` | 0.1     | Max TNF consumed per epithelial cell per step                          |
| `epi_max_il1_uptake`                               | `float` | 0.1     | Max IL-1 consumed per epithelial cell per step                         |
| `epi_t1ifn_secretion`                              | `float` | 0.75    | T1IFN secreted per triggering event (bat only)                         |
| `epi_t1ifn_secretion_prob`                         | `float` | 0.01    | Per-cell probability of baseline T1IFN secretion per step (bat only)   |
| `epi_pdamps_secretion_prob`                        | `float` | 0.01    | Per-cell probability of metabolic P/DAMP secretion per step            |
| `infected_epi_t1ifn_secretion`                     | `float` | 1.0     | T1IFN secreted by an infected cell per step                            |
| `infected_epi_il18_secretion`                      | `float` | 0.11    | IL-18 secreted by an infected cell per step                            |
| `infected_epi_il6_secretion`                       | `float` | 0.10    | IL-6 secreted by an infected cell per step (requires IL-1+TNF > 1)     |

## Endothelium

| Parameter                           | Type    | Default | Description                                                           |
|-------------------------------------|---------|---------|-----------------------------------------------------------------------|
| `activated_endo_death_threshold`    | `float` | 0.5     | TNF+IL-1 level below which activated endothelium deactivates          |
| `activated_endo_adhesion_threshold` | `float` | 36      | Adhesion counter value at which PMN spawning begins                   |
| `activated_endo_pmn_spawn_prob`     | `float` | 0.1     | Per-step probability of spawning a PMN once adhesion threshold is met |
| `activated_endo_pmn_spawn_dist`     | `float` | 5.0     | Distance PMN jumps from its spawn site                                |

## PMNs

| Parameter                    | Type    | Default | Description                                |
|------------------------------|---------|---------|--------------------------------------------|
| `pmn_max_age`                | `int`   | 36      | Steps before a PMN bursts (~6 hr lifespan) |
| `pmn_ros_secretion_on_death` | `float` | 10.0    | ROS released when a PMN bursts             |
| `pmn_il1_secretion_on_death` | `float` | 1.0     | IL-1 released when a PMN bursts            |

## NK cells

| Parameter           | Type    | Default | Description                                                        |
|---------------------|---------|---------|--------------------------------------------------------------------|
| `nk_ifng_secretion` | `float` | 1.0     | IFNg secreted per NK per step (if T1IFN > 0, IL-12 > 0, IL-18 > 0) |

## Dendritic cells

| Parameter                       | Type    | Default | Description                                                  |
|---------------------------------|---------|---------|--------------------------------------------------------------|
| `dc_t1ifn_activation_threshold` | `float` | 1.0     | T1IFN level required to activate a DC                        |
| `dc_il12_secretion`             | `float` | 0.5     | IL-12 secreted per activated DC per step                     |
| `dc_ifng_secretion`             | `float` | 0.5     | IFNg secreted per activated DC per step                      |
| `dc_il6_secretion`              | `float` | 0.4     | IL-6 secreted per activated DC per step (requires IL-1 > 1)  |
| `dc_il6_max_uptake`             | `float` | 0.1     | Max T1IFN consumed per DC per step                           |

---
 
## Grid dimensions

| Parameter     | Type  | Default | Description                |
|---------------|-------|---------|----------------------------|
| `GRID_WIDTH`  | `int` | (req.)  | Grid width (standard: 51)  |
| `GRID_HEIGHT` | `int` | (req.)  | Grid height (standard: 51) |

## Diffusion constants

Each molecular field has an independent diffusion constant controlling the fraction
that spreads to the 8 neighbors each step (NetLogo-style diffusion).

| Parameter                             | Default |
|---------------------------------------|---------|
| `extracellular_virus_diffusion_const` | 0.05    |
| `T1IFN_diffusion_const`               | 0.1     |
| `PAF_diffusion_const`                 | 0.1     |
| `ROS_diffusion_const`                 | 0.1     |
| `P_DAMPS_diffusion_const`             | 0.1     |
| `IFNg_diffusion_const`                | 0.2     |
| `TNF_diffusion_const`                 | 0.2     |
| `IL6_diffusion_const`                 | 0.2     |
| `IL1_diffusion_const`                 | 0.2     |
| `IL10_diffusion_const`                | 0.2     |
| `IL12_diffusion_const`                | 0.2     |
| `IL18_diffusion_const`                | 0.2     |
| `IL8_diffusion_const`                 | 0.3     |

## Evaporation and cleanup

After diffusion, fields are multiplied by an evaporation constant, then values
below a cleanup threshold are set to zero.

| Parameter                               | Type    | Default | Description                                          |
|-----------------------------------------|---------|---------|------------------------------------------------------|
| `evap_const_1`                          | `float` | 0.99    | Evaporation multiplier for cytokines                 |
| `evap_const_2`                          | `float` | 0.9     | Evaporation multiplier for faster-clearing mediators |
| `extracellular_virus_cleanup_threshold` | `float` | 0.05    | Virus amounts below this are zeroed                  |
| `cleanup_threshold`                     | `float` | 0.1     | Cytokine amounts below this are zeroed               |


## Array-size bookkeeping

These control the pre-allocated agent arrays. PMN arrays auto-expand when full;
the others raise an error or stop creating agents at the limit.

| Parameter         | Type   | Default | Description                                                    |
|-------------------|--------|---------|----------------------------------------------------------------|
| `MAX_PMNS`        | `int`  | 3000    | Maximum PMN array size (auto-expands if exceeded)              |
| `MAX_DCS`         | `int`  | 200     | Maximum DC array size                                          |
| `MAX_MACROPHAGES` | `int`  | 200     | Maximum macrophage array size                                  |
| `MAX_NKS`         | `int`  | 200     | Maximum NK array size                                          |
| `HARD_BOUND`      | `bool` | `True`  | Raise error if NK array bound is exceeded instead of expanding |

---

## Bat-specific parameters

These parameters exist to support the bat/human comparison from the original paper.
For human-model runs (`is_bat=False`) the `bat_*` values are unused and the
`human_*` values apply.

| Parameter                   | Type    | Human default | Bat default | Description                                               |
|-----------------------------|---------|---------------|-------------|-----------------------------------------------------------|
| `is_bat`                    | `bool`  | `False`       | `True`      | Switch bat-specific immune parameters on/off              |
| `human_endo_activation`     | `int`   | 5             | --          | TNF+IL-1 threshold to activate endothelium                |
| `bat_endo_activation`       | `int`   | --            | 10          | TNF+IL-1 threshold to activate endothelium (bat)          |
| `human_metabolic_byproduct` | `float` | 0.2           | --          | P/DAMP secreted per metabolic event                       |
| `bat_metabolic_byproduct`   | `float` | --            | 2.0         | P/DAMP secreted per metabolic event (bat)                 |
| `human_viral_lower_bound`   | `float` | 0.0           | --          | Minimum intracellular virus (allows clearance to zero)    |
| `bat_viral_lower_bound`     | `float` | --            | 1.0         | Minimum intracellular virus (persists at low level)       |
| `human_t1ifn_effect_scale`  | `float` | 0.01          | --          | T1IFN suppression scale on viral replication              |
| `bat_t1ifn_effect_scale`    | `float` | --            | 0.1         | T1IFN suppression scale on viral replication (bat)        |

The following are init-only and bat-only (used only when `is_bat=True`):

| Parameter               | Type    | Default | Description                                                |
|-------------------------|---------|---------|------------------------------------------------------------|
| `bat_t1ifn_init_amount` | `float` | 5.0     | T1IFN placed at initialisation on bat cells                |
| `bat_t1ifn_init_prob`   | `float` | 0.01    | Per-cell probability of receiving initial T1IFN (bat only) |
