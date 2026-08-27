---
title: DNDSR Reaction Experiments
date: 2026-08-27T16:00:00+08:00
type: post
categories: ['DNDSR']
tags: ['reactive-flow', 'Cantera', 'detonation', 'flame', 'positivity-preserving']
image: https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/dndsr-reaction-experiments/detonation_speed_final.png
---

# DNDSR Reaction Experiments

This post summarizes the current reactive-flow implementation in DNDSR, the conventions and pitfalls we hit while coupling Cantera chemistry to the Euler solver, and two 1-D validation cases: a stoichiometric H2/air premixed flame and a 1-D H2/O2 detonation.

## Reactive-flow implementation and pitfalls

The reactive branch is `dev/harry`. The major feature commit is [1f20f525](https://github.com/harryzhou2000/DNDSR/commit/1f20f525) ("Reactive flows, mesh hardening, core improvements, CI update, developer tooling"), which introduced the `NS_EX`/`NS_EX3D` Euler model, the `SourceTermContributor` dispatch, the Cantera-backed `ChemicalSource`, and `PhysicsProperties`. The Cantera build was added in [5e953563](https://github.com/harryzhou2000/DNDSR/commit/5e953563).

### Cantera internal energy vs. DNDSR sensible representation

Cantera reports `intEnergy_mass(T,Y)` relative to the thermochemical reference of the mechanism (typically the standard enthalpy of formation at 298.15 K). DNDSR stores **total** `rhoE` compatible with that convention:

$$
  u_{\text{sent}} = \left( \frac{\rho E_{\text{total}}}{\rho} - \frac{1}{2} |\mathbf{v}|^2 \right) U_0^2
$$

and calls Cantera's UV solver directly. There is no 298 K-to-0 K bridge.

For positivity-preserving (PP) bookkeeping DNDSR defines a **sensible** internal energy by subtracting a base-energy offset:

$$
  \rho e_{\text{sensible}} = \rho E_{\text{total}} - \frac{1}{2}\rho |\mathbf{v}|^2 - \rho E_{\text{base}}
$$

$$
  \rho E_{\text{base}} = \frac{\rho}{U_0^2} \sum_k Y_k e_{\text{base},k}
$$

where `e_{text{base},k}` is the species internal energy at a reference temperature `T_{text{base}}` (usually the minimum Cantera tabulated temperature). This offset is **only** a PP/accounting device; it is never sent to Cantera.

The pressure closure uses the bookkeeping sensible energy:

$$
  p = (\gamma_{\text{eq}} - 1) \rho e_{\text{sensible}}, \qquad
  \gamma_{\text{eq}} = 1 + \frac{p}{\rho e_{\text{sensible}}}
$$

while the acoustic speed uses the real frozen-composition ratio:

$$
  a^2 = \frac{c_p}{c_v} \frac{p}{\rho}.
$$

Keeping `$\gamma_{\text{eq}}$` (for pressure/energy) separate from `$c_p/c_v$` (for wave speeds) was a key fix in Phase 4d of the design (see `docs/dev/reactiveFlow/ReactiveFlowDesign.md`).

### Positivity preserving for species

Species are transported as `$\rho Y_k$` with the last species dependent:

$$
  Y_N = 1 - \sum_{k=1}^{N_s-1} Y_k.
$$

Two PP operations are in play:

1. **Mean-state repair (`RepairCellMeanSpecies`)** — after every update or restart, each transported `$\rho Y_k$` is clipped to a tiny floor and then scaled so that the dependent species retains a small positive margin:

```cpp
const real sumRhoYMax = rho * (1.0 - 1e-14);
if (sumRhoY > sumRhoYMax) rhoY_k *= sumRhoYMax / sumRhoY;
```

2. **Reconstruction repair (`RepairMassFractions` / `tolerantMassFractions`)** — at quadrature points the temporary composition is projected onto the simplex before it is fed to Cantera for production rates, transport properties, and enthalpies.

The important mismatch: **convective and diffusive species fluxes use the unprojected reconstructed `$\rho Y_k$`**, while **chemical source terms and thermodynamic properties use the repaired mass fractions**. This was explicitly clarified and fixed in [8e2bc446](https://github.com/harryzhou2000/DNDSR/commit/8e2bc446) ("fix(Euler): repair reactive initialization and source states").

### Source Jacobian and full-block coupling

`ChemicalContributor` evaluates the chemical source as

$$
  \dot{\omega}_k = M_k \Omega_k(T,p,Y)
$$

and fills the full source Jacobian block in matrix-block mode:

$$
  J_{ij} = M_k \frac{\partial \Omega_k}{\partial U_j}
$$

including species–species, temperature, and momentum couplings (the latter through kinetic-energy redistribution). The commit [1f20f525](https://github.com/harryzhou2000/DNDSR/commit/1f20f525) added analytic Jacobians and validated them against finite differences in `euler_test_chem_ode`.

### Mixture-induced `rhoE` flux

Species diffusion contributes to the energy flux through the total species enthalpies:

$$
  F_{\text{visc},E} = \boldsymbol{\tau} \cdot \mathbf{u} + k\nabla T - \sum_k h_k J_k
$$

with the correction velocity enforcing `$\sum_k J_k = 0$`. The heat flux is also corrected for `$\nabla R(Y)$` because the thermal conductivity term includes `$-k T/R \nabla R$`. This is implemented in `PhysicsProperties::addMixtureAveragedSpeciesDiffusionFlux` (see `src/Euler/Physics/PhysicsProperties.hpp:1082`). Using the full `$h_k$` rather than the base-energy offset is essential; using the offset would drop the formation-enthalpy contribution carried by diffusing species.

### Other state-mismatching errors

- **Base-energy increment in PP limiter**: because `rhoE_base` changes linearly with `$\rho$` and `$\rho Y_k$`, the safe compression ratio in `EvaluateCellRHSAlpha` must account for `$\Delta\rho E_{\text{base}}$`. The two-state contract uses `rhoE_base(u)` and `rhoE_base(u+inc)` explicitly; ignoring it produced non-physical `alpha` factors.
- **Initialization/restart**: early runs failed because mean species were not repaired after reading a state from JSON. [8e2bc446](https://github.com/harryzhou2000/DNDSR/commit/8e2bc446) added `RepairCellMeanState` and shared the simplex repair with the update path.
- **Mass-fraction input in CJ tools**: the CJ/detonation postprocessing initially passed mole fractions to SDToolbox when it should have passed mass fractions; fixed in [f35d6469](https://github.com/harryzhou2000/DNDSR/commit/f35d6469).

## 1-D premixed flame

The flame case is `cases/eulerEX/config_1d_premixed_stoichiometric.json`, using the `h2o2.yaml` mechanism with mixture-averaged transport. The Cantera reference flame speed is `Su = 2.2540` m/s with burned temperature 2360.02 K. The domain is 2 cm with 400 cells; the front is tracked by the midpoint-temperature crossing (`T_{text{mid}} \approx 1330` K).

| Method | dt | `Su` | vs Cantera |
|---|---|---|---|
| Non-Strang | 1e-3 | 2.46 m/s | 1.09× |
| Non-Strang | 2e-3 | 2.41 m/s | 1.07× |
| Strang | 1e-3 | 2.66 m/s | 1.18× |
| Strang | 2e-3 | 2.97 m/s | 1.32× |

The coupled (non-Strang) solver stays within 7–13 % of Cantera, while Strang splitting systematically over-predicts `Su` by 18–36 % and the error grows with `dt`.

![Flame front marker vs time](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/dndsr-reaction-experiments/front_marker_time.png)

The faint x-markers show where each flame pins at the right boundary; the squares mark the 5-point windows used for the linear speed fit.

## 1-D detonation

The detonation case is `cases/eulerEX/config_1d_detonation.json`, stoichiometric H2/O2 at 300 K and 1 atm. The Cantera CJ reference from `estimate_cj.py` is `U_{text{CJ}} = 2836.4` m/s. The domain is 5 cm with 5000 cells (`dx = 10 \mu`m`). Runs use coupled and Strang splitting with `dt = 1e-6` and `4e-6`.

| Method | dt | Simulated `U` | Error |
|---|---|---|---|
| Coupled | 1e-6 | 2842.5 m/s | +0.21 % |
| Strang | 1e-6 | 2842.5 m/s | +0.21 % |
| Coupled | 4e-6 | 2908.8 m/s | +2.55 % |
| Strang | 4e-6 | 2842.5 m/s | +0.21 % |

At the fine timestep both methods agree with Cantera within 0.21 %. At `dt = 4e-6` the coupled run is slightly faster (+2.55 %), while Strang remains at +0.21 %, suggesting the Strang splitting error is smaller than the coupled temporal truncation error for this problem at this timestep.

![Detonation shock front vs time](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/dndsr-reaction-experiments/detonation_speed_final.png)

The two `dt = 1e-6` curves overlap, confirming identical propagation speeds. The steeper `Coupled dt=4e-6` line shows the temporal discretisation error.

## Takeaway

Reactive flow in DNDSR is currently a fully-coupled, implicit, full-block-Jacobian solver with strict PP bookkeeping. The main pitfalls are not the chemistry itself but the bookkeeping around energy offsets, species simplex constraints, and which operations see repaired vs. unprojected states. The validation cases show that once those conventions are enforced, both flame and detonation speeds agree with Cantera to within a few percent.
