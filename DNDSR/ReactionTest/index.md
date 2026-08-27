---
title: DNDSR Reaction Experiments
date: 2026-08-27T17:00:00+08:00
type: post
categories: ["DNDSR"]
tags: ["DNDSR", "reactive flow", "Cantera", "positivity-preserving", "detonation", "premixed flame"]
---

This note summarizes the current reactive-flow implementation in DNDSR and the practical pitfalls we hit while validating it against 1-D premixed-flame and detonation problems. The discussion is anchored to actual commits in the DNDSR repository.

## 1. Reactive-flow implementation overview

DNDSR's reactive solver couples the compressible Euler/Navier-Stokes equations with Cantera thermodynamics, kinetics, and mixture-averaged transport. The feature landed in the large dev/harry integration ([1f20f525](https://github.com/harryzhou2000/DNDSR/commit/1f20f525a95d3d2b006193b6bfddd82efdb44ee0)). Key pieces include:

- Multi-species transport with an $N_s - 1$ independent-species formulation.
- A PIMPL ChemicalSource wrapper around Cantera (src/Euler/Chemistry/ChemicalSource.{hpp,cpp}) so that the rest of the Euler module is Cantera-free at compile time.
- Reactive Roe/HLLC/HLLEP Riemann solvers with species-aware positivity preservation.
- Optional Strang splitting (sourceStrangSplitting) versus a fully coupled (non-Strang) pseudo-time source.

The conservative state for the extended Euler model is

$$
U = \begin{bmatrix}
\rho \\
\rho u \\
\rho v \\
\rho w \\
\rho E \\
\rho Y_1 \\
\vdots \\
\rho Y_{N_s-1}
\end{bmatrix},
$$

where the last species $Y_{N_s}$ (usually N2) is algebraically dependent:

$$
\rho Y_{N_s} = \rho - \sum_{k=1}^{N_s-1} \rho Y_k .
$$

## 2. Energy bookkeeping: Cantera's internal energy vs. DNDSR's sensible representation

### 2.1 The convention gap

Cantera reports an absolute specific internal energy $u_k^{abs}(T)$ for each species. In a reacting simulation the zero point of that energy is irrelevant for the dynamics, but it can be huge and of mixed sign. To keep the total-energy conservative variable well-conditioned and to guarantee a positive sensible internal energy, DNDSR subtracts a base internal energy evaluated at a reference temperature $T_{base}$:

$$
e_{base,k} = u_k^{abs}(T_{base}) .
$$

The sensible specific internal energy is then

$$
e_{sensible} = \sum_k Y_k \bigl(u_k^{abs}(T) - e_{base,k}\bigr) .
$$

In conservative form the total energy density is split as

$$
\rho E = \underbrace{\rho e_{sensible} + \frac{1}{2}\rho |\mathbf{u}|^2}_{\text{sensible part}} + \underbrace{\rho \sum_k Y_k e_{base,k}}_{\rho e_{base}} .
$$

The volumetric base energy is

$$
\rho E_{base} = \rho \sum_k Y_k e_{base,k} .
$$

Because $e_{base,k}$ is constant, $\rho E_{base}$ depends only on the species field, and the sensible part carries the temperature.

### 2.2 Temperature positivity through sensible energy

DNDSR does **not** enforce $T > 0$ directly. Instead, every positivity-preserving check enforces

$
\rho e_{\text{sensible}} > 0.
$

Because

$
e_{\text{sensible}}(T,Y)
= \sum_k Y_k \bigl(u_k^{abs}(T) - e_{\text{base},k}\bigr)
= e_{\text{tot}}(T,Y) - e_{\text{tot}}(T_{\text{base}},Y),
$

and the mixture heat capacity at constant volume is positive, $e_{\text{sensible}}(T,Y)$ is strictly increasing with $T$. Therefore

$
\rho e_{\text{sensible}} > 0
\;\Longleftrightarrow\;
T > T_{\text{base}}.
$

That is why `AssertMeanValuePP`, `EvaluateCellRHSAlpha`, and `EvaluateURecBeta` all operate on the sensible energy rather than on temperature: requiring $T > T_{\text{base}}$ is the same physical requirement as $T$ staying above the lowest thermodynamically tabulated temperature. A cell with $T \le T_{\text{base}}$ would have non-positive sensible energy and would be rejected by the limiter.

### 2.3 Equation of state and equivalent gamma

The ideal-gas closure is written in terms of the sensible internal energy:

$$
p = (\gamma_{eq} - 1) \rho e_{sensible} .
$$

For a reactive mixture, DNDSR defines an equivalent $\gamma_{eq}$ so that this relation holds with the exact Cantera pressure $p = \rho R_{mix} T$:

$$
\gamma_{eq} = 1 + \frac{p}{\rho e_{sensible}} = 1 + \frac{R_{mix} T}{e_{sensible}} .
$$

This is implemented in PhysicsProperties::gammaEq (src/Euler/Physics/PhysicsProperties.hpp). The acoustic decomposition in the Roe solver and the primitive-to-conservative conversions in Gas.hpp all accept an optional $\rho E_{base}$ argument.

### 2.4 Temperature floor and dynamic base temperature

Early code hard-coded temperature floors at 200 K or 300 K. These magic numbers caused subtle mismatches with the base-energy bookkeeping. They were replaced by ChemicalSource::baseTemperature() ([f78b0333](https://github.com/harryzhou2000/DNDSR/commit/f78b03338acb948df266fde1a225decb1ca29f92)), so the thermodynamic floor is exactly the temperature at which the base energy is evaluated.

## 3. Positivity-preserving of species

### 3.1 Independent/dependent species and the simplex

DNDSR transports only $N_s - 1$ species. The dependent species is recovered by mass conservation. For the mean state to be physically admissible we need

$$
\rho Y_k \ge 0, \qquad \sum_{k=1}^{N_s-1} \rho Y_k \le \rho .
$$

These checks are enforced in AssertMeanValuePP (src/Euler/EulerEvaluator.hxx). If either condition fails, the cell mean is flagged as non-physical.

### 3.2 Repairing mass fractions for Cantera

Reconstructed quadrature states and boundary values may lie slightly outside the species simplex. Before calling Cantera, DNDSR repairs them with Chemistry::RepairMassFractions (src/Euler/Chemistry/ChemicalSource.hpp):

1. Clamp each transported $\rho Y_k$ to $[0, \rho]$ and divide by $\rho$ to get a trial $Y_k$.
2. Set the dependent species as $Y_{N_s} = \max\{0, 1 - \sum_{k=1}^{N_s-1} Y_k \}$.
3. Renormalize the whole vector so that $\sum_k Y_k = 1$.

Crucially, the repair only touches the temporary composition passed to Cantera. The transported conservative variables $\rho Y_k$ are left untouched; otherwise the solver would quietly inject/consume mass during reconstruction.

### 3.3 Reconstruction PP vs. mean-state PP

There are two different positivity-preserving constraints and they are not the same:

- **Mean-state PP** operates on the cell-average $U_i$. It asserts $\rho Y_k \ge 0$ and $\sum \rho Y_k \le \rho$, and checks that the sensible energy is above the base energy.
- **Reconstruction PP** operates on the reconstructed quadrature values $U_{iG}$. In EvaluateURecBeta it checks that the species block is non-negative and that the row-wise sum satisfies $\sum_k \rho Y_{k,G} \le \rho_G$ before accepting the reconstruction.

The reconstruction check was made species-aware in [ec3eb0ba](https://github.com/harryzhou2000/DNDSR/commit/ec3eb0ba83926bed04a220a00d7281e2cfd363f7).

## 4. Other pitfalls: state mismatching, source Jacobian, and rhoE flux

### 4.1 Clipped vs. raw base energy — the central mismatch

PhysicsProperties provides two base-energy evaluations:

- mixtureBaseInternalRhoE(U) — calls massFractionsVector(U), which repairs negative/overshoot species before computing $\rho E_{base}$.
- mixtureBaseInternalRhoERaw(U) — uses the raw $\rho Y_k$ values directly.

The Raw version is linear in $U$, so it is attractive for reconstruction algebra. However, gammaEq and the Riemann solver use the clipped version. This created a mismatch: the PP limiter could declare a reconstructed state valid using Raw, but gammaEq would compute a non-positive sensible energy because the clipped composition increased $\rho E_{base}$. The resulting crash was "$e_{sensible} \le 0$" inside gammaEq.

[ec3eb0ba](https://github.com/harryzhou2000/DNDSR/commit/ec3eb0ba83926bed04a220a00d7281e2cfd363f7) fixed this by switching the PP-sensitive calls in CompressInc, CompressRecPart, EvaluateCellRHSAlpha, AssertMeanValuePP, AddFixedIncrement, and EvaluateURecBeta to the clipped mixtureBaseInternalRhoE. EvaluateURecBeta also gained a post-$\theta_P$ guard that recomputes the sensible energy with the clipped base energy and forces a full fallback if it is still invalid.

### 4.2 Barth slope limiter and energy decay

Species columns also participate in the gradient limiter. [fc4cd0e5](https://github.com/harryzhou2000/DNDSR/commit/fc4cd0e56eaa3ea6babc59f6c7af4ff060254a1e) corrected a bug in LimiterUGrad where the maximum neighboring state was not accumulated correctly, and added an iterative decay loop that reduces the gradient until both sensible-energy and species positivity are satisfied. The limiter now uses the clipped base energy consistently with gammaEq.

### 4.3 Source Jacobian term

In SourceTermContributor.hpp, the ChemicalContributor evaluates the chemical source in Mode 0 (residual) but explicitly asserts false in Mode 1 (diagonal source Jacobian):

> "ChemicalContributor: diagonal-Jacobian mode not implemented."

For implicit time stepping the reactive source is therefore treated as explicitly or embedded inside the ODE integrator (Strang) rather than being linearized into the Newton/Krylov Jacobian. This is a current limitation for very stiff chemistry.

### 4.4 Mixture-induced $\rho E$ flux and gradient correction

Because $\rho E_{base}$ varies with composition, the pressure gradient derived from the EOS must be corrected for $\nabla(\rho E_{base})$. Gas::GradientCons2Prim_IdealGas does this:

$$
\nabla p = (\gamma_{eq} - 1) \Bigl( \nabla(\rho E) - \frac{1}{2}\nabla(\rho |\mathbf{u}|^2) - \nabla(\rho E_{base}) \Bigr),
$$

with

$$
\nabla(\rho E_{base}) = \sum_{k=1}^{N_s-1} (e_{base,k} - e_{base,N_s}) \nabla(\rho Y_k) + e_{base,N_s} \nabla \rho .
$$

The inviscid flux itself already carries the total energy through the mass flux, but the gradient/viscous paths need this explicit correction or the pressure and temperature fields become inconsistent where species gradients are strong.

### 4.5 Initialization and restart repairs

[8e2bc446](https://github.com/harryzhou2000/DNDSR/commit/8e2bc4466b43cacee911117bccd8bf0ea2539569) made initialization and restart loading apply the same simplex repair that is used during time stepping. Without this, interpolated/extrapolated initial fields could enter the first RHS evaluation with negative species masses and crash immediately. A focused MPI regression test (test_EulerEvaluatorReactive.cpp) now covers this at 1, 2, 4, and 8 ranks.

## 5. 1-D premixed H2/air flame test

The flame case uses the stoichiometric H2/air mechanism h2o2.yaml with mixture-averaged transport. Cantera gives a reference laminar flame speed $S_u = 2.2540$ m/s and a burned temperature of 2360.02 K.

Configuration highlights:

- Mesh: Uniform_01_400.cgns, 400 cells, 2 cm domain, $\Delta x = 0.05$ mm.
- BCIn on both ends: burned products on the left, unburned reactants on the right.
- Initial: tanh profile centered at $x = 5$ mm, width 1 mm.
- Time integrator: ESDIRK2, CFL 10, 200 pseudo-time steps per physical step.
- Limiter: PP reconstruction limiter enabled from step 0.
- MPI: 8 ranks.

The front is tracked by the midpoint-temperature crossing ($T_{mid} \approx 1330$ K), and the flame speed is recovered as

$$
S_u = \dot{x}_{front} - u_{unburned},
$$

because the BCIn boundaries do not represent a free flame.

![Front marker positions for the six flame runs](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/dndsr-reaction-experiments/front_marker_time.png)

### Results

| Method | $\Delta t$ | $S_u$ | vs. Cantera |
|--------|------------|-------|-------------|
| Non-Strang | 1e-3 | 2.46 m/s | 1.09× |
| Non-Strang | 2e-3 | 2.41 m/s | 1.07× |
| Non-Strang R1 | 2e-3 | 2.54 m/s | 1.13× |
| Strang | 1e-3 | 2.66 m/s | 1.18× |
| Strang | 2e-3 | 2.97 m/s | 1.32× |
| Strang R1 | 2e-3 | 3.06 m/s | 1.36× |

Observations:

- The fully coupled (non-Strang) scheme agrees with Cantera to within 7–13 % on this coarse mesh.
- Strang splitting systematically over-predicts $S_u$, and the error grows with $\Delta t$, consistent with an $O(\Delta t^2)$ splitting error.
- Repeat runs are consistent, but the 2 cm domain is too short; the fit window is restricted to $t_{code} < 0.6$ to avoid boundary influence.

## 6. 1-D H2/O2 detonation test

The detonation case uses the same h2o2.yaml mechanism. A Cantera CJ analysis gives $U_{CJ} = 2836.4$ m/s with an induction length of about 50 $\mu$m.

Configuration highlights:

- Mesh: Uniform_01_5000.cgns, 5000 cells, 5 cm domain, $\Delta x = 10$ $\mu$m (5 cells across the induction zone).
- Left wall: BCWallInvis; right inflow: BCIn with unburned H2/O2 at rest.
- Initial spark: $T = 3500$ K, $p = 20$ bar in $x < 1$ mm.
- MPI: 16 ranks.

![Shock-front position vs. time for the four detonation runs](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/dndsr-reaction-experiments/detonation_speed_final.png)

### Results

| Method | $\Delta t$ | Simulated $U$ | Error vs. CJ |
|--------|------------|---------------|--------------|
| Coupled | 1e-6 | 2842.5 m/s | +0.21 % |
| Strang | 1e-6 | 2842.5 m/s | +0.21 % |
| Coupled | 4e-6 | 2908.8 m/s | +2.55 % |
| Strang | 4e-6 | 2842.5 m/s | +0.21 % |

Observations:

- At the fine time step both coupled and Strang reproduce the CJ speed to better than 0.25 %.
- At $\Delta t = 4\times10^{-6}$ the coupled scheme drifts +2.55 %, while Strang stays at +0.21 %. For this problem the Strang splitting error is smaller than the coupled temporal truncation error at large steps.
- Strang is also cheaper (2h47m vs. 3h05m wall time at $\Delta t = 4\times10^{-6}$).
- The ZND structure is clearly visible: shock front → induction zone → reaction zone → expansion products.

## 7. Takeaways

1. The sensible-energy split with a species-dependent $\rho E_{base}$ is essential for positivity-preserving reactive flow, but it introduces a clipped vs. raw bookkeeping mismatch that must be tracked consistently through the limiter, the Riemann solver, and the source term.
2. Species positivity has two stages: mean-state simplex assertions and reconstruction quadrature checks. Repairing mass fractions should only touch temporary buffers passed to Cantera.
3. The reactive source Jacobian is not yet assembled for implicit linear solvers; stiff chemistry still relies on the ODE/Strang path.
4. On the tested 1-D problems the implementation is accurate enough for engineering validation (flame speed within 10 %, detonation speed within 0.25 %), but the choice between coupled and Strang is problem-dependent rather than uniformly superior.
