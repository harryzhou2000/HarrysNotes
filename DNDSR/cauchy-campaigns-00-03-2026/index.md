---
title: "Cauchy campaigns 00–03: faster updates, converged solves, and a rotating vortex"
slug: cauchy-campaigns-00-03-2026
date: 2026-11-13T11:30:00+08:00
type: post
draft: false
description: "Euler/current/old Cauchy comparisons, 168 converged multigrid solves, fixed-work DMR timing, and a full-return sliding-interface vortex study."
categories: ["DNDSR"]
tags: ["CFD", "Cauchy", "multigrid", "performance", "sliding interface"]
image: https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/augustin-louis-cauchy.jpg
math: true
---

Four campaigns now connect Cauchy's update cost to converged flow solutions and a moving-interface accuracy test. All twelve final campaign jobs completed successfully, including **168 converged steady solves** and three tight-tolerance vortex runs through a full return at \(t=10\).

*Cover: Augustin-Louis Cauchy, photographed by Charles Reutlinger. Public-domain portrait from the Smithsonian Libraries via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Augustin_Louis_Cauchy.jpg).*

Current Cauchy is faster than Euler in the reported timing comparisons. Against old Cauchy, the outcome depends on the case and boundary treatment. The vortex study also shows why inner convergence belongs in an accuracy assessment: the original loose-tolerance sequence had nonmonotone errors, while the tighter repeat produced a decreasing sequence consistent with the intended fourth-order coupled method over the three tested meshes.

These results describe the frozen campaign revisions. They are not measurements of the later source consolidation, and the full DNDSR regression suite is outside this post's scope.

## Four complementary tests

| Campaign | Question | Scope |
|---|---|---|
| 00 | Do spatial residuals agree, and how much do repeated updates cost? | 64 ranks; repeated batches of 32 nonlinear corrections |
| 01 | Does faster work translate into faster converged solutions? | Cylinder and NACA, seven policies, three solver versions, four boundary/overlap settings |
| 02 | What is the cost at exactly matched DMR update counts? | One physical step, five implicit stages, 500 updates per solver |
| 03 | Does the moving two-domain vortex recover its intended accuracy? | A full return on three meshes, with loose and tight nonlinear tolerances |

Each job used one scheduler-selected node, with one thread for OpenMP and the numerical libraries. Compared solvers ran serially within each matrix allocation. Overlap-on/off and boundary-Jacobian jobs used separate allocations, so their raw timing differences cannot isolate either option's effect.

## 00: spatial agreement does not imply identical nonlinear updates

The initial spatial right-hand sides agree to about \(1.70\times10^{-12}\) relative error, and the reset-repeat checks passed. After 32 nonlinear corrections, however, the largest component-relative state deviation is 0.141277 with physical boundary derivatives and 0.0876419 with the Euler-scalar boundary option. The latter reduces the mismatch without establishing whole-solve identity. Overlap-on/off schedules give the same recorded comparison vectors.

The timing below is the median number of seconds per **32-update batch**. Each trial averages ten batches using the maximum-rank elapsed time; five trials were measured. Reset and setup are excluded.

| Boundary Jacobian / overlap | Cauchy [s] | Euler [s] | Cauchy/Euler |
|---|---:|---:|---:|
| Physical / on | 1.440194 | 1.955659 | 0.736 |
| Physical / off | 1.303894 | 1.740597 | 0.749 |
| Euler-scalar / on | 1.388495 | 1.759128 | 0.789 |
| Euler-scalar / off | 1.304466 | 1.728166 | 0.755 |

The physical option includes the frozen exterior-state derivative; Euler-scalar recovers Euler's scalar boundary diagonal in the comparison path. Production defaults remain physical. Matching the stationary linear-pass counts did not remove the update differences; their remaining cause has not been isolated.

## 01: all 168 steady solves converge, with mixed gains against the old solver

Four matrices each contain 42 solves: Euler, current Cauchy, and old Cauchy across seven policies and two cases. All rows converged. The inviscid cylinder uses 16 ranks and a density tolerance of \(10^{-10}\); the NACA SA-RANS case uses 64 ranks and \(10^{-6}\). Both use unlimited variational reconstruction. The policies span single-grid LU-SGS/ILU, one or two coarse levels, and GMRES+ILU.

The ratios below are medians of seven policy-wise **whole-process** time ratios. Values below one favor current Cauchy. Setup, output, and differing convergence counts are included.

| Boundary Jacobian | Overlap | Case | Current/Euler | Current/old |
|---|---|---|---:|---:|
| Physical | on | Cylinder | 0.573 | 0.806 |
| Physical | on | NACA | 0.756 | 1.095 |
| Physical | off | Cylinder | 0.566 | 0.806 |
| Physical | off | NACA | 0.756 | 1.110 |
| Euler-scalar | on | Cylinder | 0.807 | 1.289 |
| Euler-scalar | on | NACA | 0.734 | 1.124 |
| Euler-scalar | off | Cylinder | 0.798 | 1.311 |
| Euler-scalar | off | NACA | 0.754 | 1.120 |

![Whole-process Cauchy timing ratios across the seven tested policies](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/mg-time-ratios.png)

*Policy-wise whole-process ratios compare current Cauchy with Euler and old Cauchy. The physical cylinder path improves on old Cauchy by about 19% at the median ratio; NACA remains about 10–11% slower. Euler-scalar cylinder is about 29–31% slower than old Cauchy. The case dependence matters more than a single aggregate speed claim.*

For a common ILU setup with four coarse nonlinear smoothing updates and overlap enabled, the times and correction counts make the convergence contribution visible. “Four ILU” refers to smoothing updates, not polynomial degree.

| Boundary / case | Euler seconds / corrections | Current seconds / corrections | Old seconds / corrections |
|---|---:|---:|---:|
| Physical / cylinder | 31.78 / 2395 | 15.65 / 1401 | 20.39 / 2467 |
| Physical / NACA | 201.61 / 981 | 148.89 / 932 | 137.45 / 957 |
| Euler-scalar / cylinder | 31.26 / 2395 | 26.45 / 2481 | 20.29 / 2467 |
| Euler-scalar / NACA | 181.47 / 981 | 129.87 / 894 | 124.81 / 957 |

![Physical-boundary cylinder residual and timing histories](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/convergence-physical-cylinder-ilu-4ilu.png)

*Cylinder, physical boundary Jacobian, ILU with four coarse smoothing updates: current Cauchy reaches the density criterion with substantially fewer corrections than either control. Residual curves use one shared normalization across the three solvers.*

![Physical-boundary NACA residual and timing histories](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/convergence-physical-naca-ilu-4ilu.png)

*NACA with the same policy: current Cauchy is faster than Euler but slower in whole-process time than old Cauchy, despite requiring slightly fewer corrections. Update throughput and convergence count must be assessed together.*

![Euler-scalar cylinder residual and timing histories](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/convergence-scalar-cylinder-ilu-4ilu.png)

*With Euler-scalar boundaries, cylinder correction counts become similar across the solvers. The large physical-boundary convergence advantage does not transfer to this setting.*

![Euler-scalar NACA residual and timing histories](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/convergence-scalar-naca-ilu-4ilu.png)

*The Euler-scalar NACA comparison retains a current-Cauchy advantage over Euler, while old Cauchy remains faster in total time. Each panel preserves the native logging cadence.*

### Reading the residuals correctly

The common absolute density norm for these steady equations is

$$
R_\rho=\sum_{i\in\mathrm{owned}}V_i|r_{\rho,i}|.
$$

Euler volume-weights its per-volume defect; Cauchy sums an already cell-integrated defect. Multiplying Cauchy's defect by volume again would change the norm. The MPI sum includes owned cells.

Plots divide the absolute norm by **one shared maximum** per comparison. Stopping instead uses each solver's own running maximum across all evaluations, including unlogged ones. Equal tolerances therefore need not give equal absolute endpoints or equal plotted heights. For the physical cylinder example, final absolute norms are \(1.9235318\times10^{-10}\), \(1.5600994\times10^{-10}\), and \(1.4380128\times10^{-10}\) for Euler, current Cauchy, and old Cauchy; each is close to its own stopping target. Euler's step-end summary is excluded from iteration curves, avoiding an artificial low initial point.

## 02: fixed-work double-Mach-reflection throughput

The DM240U diagnostic runs Euler and Cauchy serially on 64 ranks. Each performs one ESDIRK4 physical step at \(\Delta t=0.0004\), with five implicit stages and 100 LU-SGS updates per stage. Both use 3WBAP and single-grid LU-SGS.

| Measurement | Euler | Cauchy |
|---|---:|---:|
| Whole process [s] | 216.8227 | 142.3615 |
| Mean adjacent-update interval [s] | 0.410960 | 0.261196 |
| Median adjacent-update interval [s] | 0.399767 | 0.257637 |
| P5–P95 interval [s] | 0.397987–0.452083 | 0.256870–0.281796 |

Cauchy's whole-process time is **65.66% of Euler's**, a reduction of about 34%. The update statistics cover 495 within-stage adjacent intervals: the first-update start anchor in each stage was not logged. Stage transitions are excluded. These numbers describe fixed work, not a converged developed DMR solution or long-time shock stability; no flow image is used to suggest otherwise.

## 03: a rotating two-domain vortex completes its return

The outer domain and rotating disk use separate P1 Delaunay meshes, P3 reconstruction, sliding point interpolation, and nonlinear interface refresh. The renderer applies the recorded rigid pose to reference-frame geometry, so the density and mesh panels move consistently.

Mesh spacing and time step refine together, with nominal \(h=10/N\) and \(\Delta t=1/N\). This measures the coupled spatial/temporal path. The initial sequence used a nonlinear tolerance of \(10^{-3}\) and cap 100; its errors were nonmonotone. Repeating with \(10^{-8}\) and cap 400, while retaining the other numerical choices, gives:

| N | Time step | Steps to t=10 | Stages per domain | Maximum corrections | Outer density L2 | Disk density L2 |
|---|---:|---:|---:|---:|---:|---:|
| 60 | 0.01666667 | 600 | 3000 | 54 | 6.7754331e-6 | 4.6958321e-5 |
| 80 | 0.01250000 | 800 | 4000 | 53 | 1.8286710e-6 | 1.0130571e-5 |
| 100 | 0.01000000 | 1000 | 5000 | 52 | 7.7557069e-7 | 3.7782606e-6 |

Every stage in the tight repeat converged. Accepted density and pressure stayed positive, with reported minima over both domains above 0.49395 and 0.37257. The three jobs took approximately 43, 72, and 115 minutes.

![Full-return vortex errors with both loose and tight nonlinear tolerances](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv-accuracy.png)

*Full-return error versus resolution: retaining the original loose-tolerance curves shows why that sequence could not support an accuracy-order inference. Tightening the nonlinear solves restores decreasing errors.*

![Tight-tolerance-only vortex accuracy comparison](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv-accuracy-tight-only.png)

*For the tight repeat, pairwise outer density L2 slopes are 4.553 and 3.844; disk slopes are 5.331 and 4.420. These three samples are consistent with the intended fourth-order coupled method, but do not isolate spatial order or prove an asymptotic rate.*

### Watch the density field and mesh motion

The videos below show the **original loose-tolerance sequence (\(10^{-3}\))**, not the tight accuracy repeat. Each contains 200 visualization times from \(t=0.05\) to 10. The posters show the final time. The disk rotates during the trajectory even though the vortex completes a return.

<figure>
<video controls playsinline preload="metadata" style="display:block;width:100%;height:auto" poster="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv60-tol1e-3-poster.png" aria-label="N equals 60 loose-tolerance vortex density and mesh motion">
<source src="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv60-tol1e-3.mp4" type="video/mp4">
<a href="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv60-tol1e-3.mp4">Download the N=60 video</a>.
</video>
<figcaption>N=60, loose tolerance: synchronized density and mesh-motion views over the full return. The density uses jet coloring; the mesh view makes the sliding-interface motion visible.</figcaption>
</figure>

<figure>
<video controls playsinline preload="metadata" style="display:block;width:100%;height:auto" poster="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv80-tol1e-3-poster.png" aria-label="N equals 80 loose-tolerance vortex density and mesh motion">
<source src="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv80-tol1e-3.mp4" type="video/mp4">
<a href="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv80-tol1e-3.mp4">Download the N=80 video</a>.
</video>
<figcaption>N=80, loose tolerance: the intermediate mesh follows the same recorded trajectory. Apparent visual smoothness is not a substitute for the tight-repeat error measurements.</figcaption>
</figure>

<figure>
<video controls playsinline preload="metadata" style="display:block;width:100%;height:auto" poster="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv100-tol1e-3-poster.png" aria-label="N equals 100 loose-tolerance vortex density and mesh motion">
<source src="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv100-tol1e-3.mp4" type="video/mp4">
<a href="https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/iv100-tol1e-3.mp4">Download the N=100 video</a>.
</video>
<figcaption>N=100, loose tolerance: the finest animated case, with the same visualization times and synchronized field/geometry poses. These frames illustrate motion, rather than certify the accuracy of the original loose solve.</figcaption>
</figure>

### Separate snapshots from the tight N=60 repeat

![Tight-tolerance N60 vortex at t=0.05](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/tight60-000000.png)

*Tight repeat, N=60, t=0.05: an early snapshot from the accuracy run. The density scale is fixed across these three frames; the geometry uses the recorded rigid pose.*

![Tight-tolerance N60 vortex at t=5.05](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/tight60-000001.png)

*Tight repeat, N=60, t=5.05: the intermediate state and actual disk orientation, rendered separately from the loose-sequence videos.*

![Tight-tolerance N60 vortex at t=10](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/cauchy-campaigns-00-03/tight60-000002.png)

*Tight repeat, N=60, t=10: the completed return used for the coarsest tight-run error entry. Completion and positive accepted states accompany, rather than replace, the quantitative error assessment.*

## What the campaigns establish

The strongest combined result is that Cauchy can reduce cost relative to Euler while converging the tested steady systems and completing the tightly solved moving-interface sequence. Old Cauchy remains a useful performance control: the current implementation improves the physical-boundary cylinder case but still regresses on several other combinations.

The next comparisons should retain correction counts alongside timings, investigate the remaining nonlinear-update mismatch, and refine time step independently of mesh spacing in the vortex test. The fixed-work DMR diagnostic supplies a throughput baseline; developed shock-flow accuracy and long-time stability require their own tests.
