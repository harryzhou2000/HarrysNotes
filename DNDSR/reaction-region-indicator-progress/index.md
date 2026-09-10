---
title: "Reaction-region indicator: a progress update"
slug: reaction-region-indicator-progress
date: 2026-09-10T16:00:00+08:00
type: post
categories: ["DNDSR"]
tags: ["reacting flow", "operator splitting", "detonation"]
image: cover.png
---

When chemistry, diffusion, and shocks occupy very different parts of a flow, a single integration treatment can be an awkward compromise. This work explores a cell-local *reaction-region indicator* that chooses how much of each cell is treated with tightly coupled chemistry and flow, and how much is handled by Strang splitting.

The goal is deliberately practical: use coupling where the interaction is strongest, while retaining the benefits of splitting where it is appropriate. It is a numerical-method selector, not a claim to identify a universal physical reaction zone.

## The idea

The selector combines three dimensionless signals:

- local chemical activity;
- local diffusion activity; and
- a shock gate based on the pressure jump.

Chemistry and diffusion must both be active before the coupled fraction becomes large. The shock gate suppresses coupling at a strong detonation shock, where splitting can be preferable for propagating the discontinuity. The result is a continuous field rather than a binary mask.

## Mathematical definition

Let \(\Delta t_{\mathrm{phys}}=\Delta t_{\mathrm{code}}L_0/U_0\) be the physical duration of a solver step. For cell \(i\), the chemical activity is

$$
a_i = \Delta t_{\mathrm{phys}}r_{\mathrm{chem},i},
\qquad
r_{\mathrm{chem},i} =
\left[
\sum_{k=1}^{N_s}
\left(\frac{\dot\omega_{k,i}W_k}{\rho_i}\right)^2
+\left(\frac{|\dot q_i|}{\rho_i c_{v,i}T_{\mathrm{scale},i}}\right)^2
\right]^{1/2}.
$$

Here \(\dot\omega_k\) is the net molar production rate, \(W_k\) the molecular weight, \(\rho\) the density, \(c_v\) the constant-volume specific heat, and

$$
\dot q_i=-\sum_{k=1}^{N_s}\dot\omega_{k,i}\bar h_{k,i},
\qquad
T_{\mathrm{scale},i}=\max(T_i,T_{\mathrm{floor}}).
$$

Thus \(r_{\mathrm{chem}}\) is a non-negative inverse chemical time scale: it combines species conversion with the fractional temperature-source rate, so neither can cancel the other.

The diffusion activity is

$$
b_i=\Delta t_{\mathrm{phys}}\frac{D_{\max,i}}{L_{\mathrm{grad},i}^{2}},
$$

where \(D_{\max,i}\) is the largest mixture species diffusivity and the local gradient length is reconstructed from face neighbours:

$$
L_{\mathrm{grad},i}^{-1}
=\max_{j\in\mathcal N(i)}
\left[
\left(
\frac{T_j-T_i}{d_{ij}\max(T_i,T_j,T_{\mathrm{floor}})}
\right)^2
+\sum_{k\in\mathcal A_{ij}}
\left(
\frac{Y_{k,j}-Y_{k,i}}
{d_{ij}\max(Y_{k,i},Y_{k,j},10^{-3})}
\right)^2
\right]^{1/2},
$$

$$
L_{\mathrm{grad},i}
=\max\!\left(\frac{1}{L_{\mathrm{grad},i}^{-1}},\Delta x_i\right).
$$

\(\mathcal N(i)\) is the face-neighbour set, \(d_{ij}\) is the distance between cell centres, and \(\mathcal A_{ij}\) retains species whose larger adjacent mass fraction is at least \(10^{-3}\). The lower bound \(\Delta x_i\) prevents a subcell gradient length.

The pressure-jump sensor and its shock gate are

$$
h_i=\max_{j\in\mathcal N(i)}
\frac{|p_j-p_i|}{\max(|p_i|,|p_j|,p_\epsilon)},
\qquad
g_h(h_i)=\frac{1}{1+(h_i/h_0)^4}.
$$

The three inputs \(a_i\), \(b_i\), and \(h_i\) are dimensionless. In particular, \(g_h\approx1\) in a smooth flame and tends to zero across a sufficiently strong pressure jump.

For positive thresholds \(a_0\), \(b_0\), and exponent \(p\), each activity is saturated by

$$
\operatorname{sat}(z;z_0,p)
=\frac{\bigl(\max(z,0)/z_0\bigr)^p}
{1+\bigl(\max(z,0)/z_0\bigr)^p},
$$

and the coupled score is

$$
C_i=
\operatorname{sat}(a_i;a_0,p)
\operatorname{sat}(b_i;b_0,p)
g_h(h_i).
$$

The original saturation is recovered by \((a_0,b_0,p)=(1,1,1)\), for which \(\operatorname{sat}(z)=z/(1+z)\).

## From score to split treatment

For the figures below, the displayed coupled fraction is the complement of the Strang fraction:

$$
c = 1 - \chi,
\qquad
\chi = 0\ \text{for fully coupled integration},
\qquad
\chi = 1\ \text{for Strang splitting}.
$$

The original logistic map is

$$
c_i^*=\sigma\!\left(
\frac{C_i-C_0-w\ln b_s}{w}
\right),
\qquad
\sigma(z)=\frac{1}{1+e^{-z}},
$$

where \(C_0\) is the score midpoint, \(w\) the transition width, and \(b_s\) a Strang-preference factor. The historical settings used \(C_0=0.005\), \(w=0.001\), \(h_0=0.08\), and \(b_s=1\).

The compact-tail Hill alternative is

$$
c_i^*=\frac{C_i^n}{C_i^n+(C_0b_s)^n},
$$

with exponent \(n>0\). Unlike the logistic map, it gives exactly \(c_i^*=0\) when \(C_i=0\). Optional face-neighbour expansion then applies

$$
c_i^{(m+1)}=
\operatorname{clip}_{[0,1]}
\left[
\max\!\left(
c_i^{(m)},
\eta\,g_h(h_i)\max_{j\in\mathcal N(i)}c_j^{(m)}
\right)
\right],
$$

where \(\eta\in[0,1]\) is the retention per pass. The tuned comparison used \(n=3\), two passes, and \(\eta=0.65\).

Finally, with \(\chi_i^*=1-c_i^*\), endpoint tolerances enforce exact limiting treatments:

$$
\chi_i=
\begin{cases}
0, & \chi_i^*\le\epsilon_c,\\
1, & 1-\chi_i^*\le\epsilon_s,\\
\chi_i^*, & \text{otherwise}.
\end{cases}
$$

The thresholds and powers change the finite-step transition, not the physical state. Since \(a_i\) and \(b_i\) both scale with \(\Delta t_{\mathrm{phys}}\), \(\chi\) is intentionally a time-step-dependent *numerical-method selector*.

## Mixed integrator

Write the semidiscrete reactive-flow system as \(\dot u=F(u)+S(u)\), with \(F\) the non-reactive flow operator and \(S\) the chemical source. The mixed method uses the same frozen \(\chi_i\) field in every part of one physical step:

$$
u^{n+1}=
B_{\chi S}\!\left(\frac{\Delta t}{2}\right)
A_{F+(1-\chi)S}(\Delta t)
B_{\chi S}\!\left(\frac{\Delta t}{2}\right)u^n.
$$

In words: each cell receives its \(\chi_i\)-weighted chemistry in two source-only half steps, while the full implicit flow step carries the complementary \((1-\chi_i)S_i\) source contribution. The selector is evaluated from the physical-step entry state and remains fixed through both half steps and all ODE stages.

## What the profiles show

The one-dimensional profiles below give an intuitive view. In flames, the coupled region follows the thin part of the thermal and chemical structure. In detonations it stays much narrower and does not simply cover the leading shock. Increasing the chosen time step broadens the selected region, which is expected: the indicator is responding to the numerical challenge of taking a larger step through a reactive, diffusive layer.

![Frozen-profile response of the coupled fraction as the numerical step changes](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/one_dimensional_profiles.png)

This is a useful distinction. The underlying chemical and diffusion rates are physical diagnostics, whereas the final selector is allowed to depend on the step size because it decides between numerical treatments. These plots keep the flow state fixed, so they reveal the selector's behavior but do not by themselves prove trajectory accuracy.

### One-dimensional a-priori profiles

The fixed-profile view is also useful at the level of the individual indicator ingredients. The flame and detonation panels below evaluate the same formula before a mixed trajectory is evolved. They answer a deliberately narrow question: where would the selector request coupled treatment for this stored thermochemical structure?

![A-priori H2/O2 flame indicator profile](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/1d-apriori-flame.png)

*Figure — Fixed H2/O2 flame profile at \(\Delta t_{\mathrm{code}}=2\times10^{-3}\). From top to bottom: temperature and pressure; major species; radical mass fractions; the chemical and diffusive one-step activities \(a\) and \(b\); their saturated factors, shock gate, and coupled score; and the final Strang fraction \(\chi\) with its coupled complement \(1-\chi\). The narrow coupled band is aligned with the flame transition. This is an a-priori classification of a fixed profile, not a propagation result.*

![A-priori H2/O2 detonation indicator profile](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/1d-apriori-detonation.png)

*Figure — Fixed H2/O2 detonation profile at \(\Delta t_{\mathrm{code}}=4\times10^{-6}\). The upper panels show the shock-adjacent thermochemical transition and radicals; the next panels show \(a\), \(b\), saturation, the pressure-jump gate, and the resulting score. Although chemical activity rises near the leading structure, the shock gate collapses at the discontinuity, leaving the displayed split fraction overwhelmingly Strang-dominant. This is a frozen-state diagnostic, not an accuracy assessment.*

### One-dimensional a-posteriori profiles

The next two figures are different evidence: they are taken from completed mixed simulations. Each compares the nontrivial logistic and tuned selector fields on their own evolved states, so it shows the selector as it actually participated in the calculation. It must not be read as an independent test of the fixed-state calibration.

![A-posteriori mixed indicator profiles for a coarse-step ESDIRK2 flame](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/1d-posterior-flame-esdirk2-dt4e-3.png)

*Figure — A-posteriori ESDIRK2 flame profiles at \(\Delta t=4\times10^{-3}\), relative to the propagated front. The logistic result occupies the top row and the tuned result the bottom row. Left panels give temperature and pressure, centre panels give the resulting chemical and diffusive activities, and right panels give the shock sensor, coupled score, and \(\chi\). Both mixed solutions retain a compact coupled interval around the flame transition, despite the different final states. This is a posterior record from a completed coarse-step calculation; it demonstrates behavior of the mixed treatments, not convergence.*

![A-posteriori mixed indicator profiles for a U2R1 detonation](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/1d-posterior-detonation-u2r1-dt8e-6.png)

*Figure — A-posteriori U2R1 detonation profiles at \(\Delta t=8\times10^{-6}\), relative to the propagated shock. Logistic and tuned treatments again occupy the upper and lower rows. Activity concentrates upstream of the discontinuity, while the shock sensor and score are sharply localized at the front; \(\chi\) remains close to one over almost the entire plotted length. The plot makes the intended detonation behavior visible: mixed integration is localized rather than a blanket replacement of Strang splitting. It does not establish mesh or inner-solver convergence.*

## A two-dimensional detonation picture

The same ingredients were examined on a cellular detonation snapshot. The temperature field has a wrinkled reaction front; the coupled-fraction panel follows that front rather than filling the entire hot product region. The shock gate makes the leading discontinuity visibly different from the reactive structure behind it.

![Temperature, chemistry, diffusion, shock gate, and coupled fraction on a cellular detonation snapshot](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/cellular_detonation_selector.png)

*Figure — Five stacked cell-field maps for the O4-Strang cellular-detonation snapshot. From top to bottom: temperature, chemical saturation, diffusion saturation, the pressure-jump shock gate, and the resulting coupled fraction \(1-\chi\). White isolines follow the cellular structure. The bright coupled band lies behind the shock rather than filling the hot products; its nonzero trace near a pressure jump is why this aggressive setting remains a demonstration rather than a default.*

The particular setting shown here is intentionally aggressive, so it is best read as a spatial demonstration rather than a recommended default. It also exposes the remaining tuning question: a useful selector must retain a sharp response to shocks while giving a sufficiently smooth transition around reaction layers.

The comparison below puts several threshold-and-power choices on the same cellular-detonation snapshot. The legacy and moderate settings remain almost entirely split at this step; the broader candidates activate along the corrugated reaction front. It is a helpful visual reminder that tuning controls *where* coupled integration is requested, not just the peak value of a scalar score.

![Two-dimensional comparison of five selector candidates on the same cellular detonation](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/cellular_detonation_candidate_comparison.png)

*Figure — Coupled fraction \(1-\chi\) for five threshold-and-power candidates at the same physical step and on the same \(x>0\) cellular-detonation field. The legacy, moderate, and conservative candidates are visually near zero; the balanced and maximum-feasible settings light up a narrow corrugated front. The shared color scale makes the contrast a selector choice, not a change in the underlying flow state.*

Time step matters in two dimensions as well. Holding the snapshot fixed, the legacy selector requests essentially no coupling at a very small step, remains near Strang splitting at the recorded base step, and becomes much more expansive at an intentionally extreme step. This sensitivity is expected for a method selector, but it also makes clear why threshold choices must be assessed together with the temporal discretization.

![Two-dimensional fixed-snapshot selector response from a small to an extreme time-step factor](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/cellular_detonation_timestep_sensitivity.png)

*Figure — One fixed O4-Strang temperature field at left, followed by the legacy coupled fraction \(1-\chi\) at \(\Delta t/\Delta t_0=10^{-2}\), \(1\), and \(10^3\). The near-uniform dark middle panels are the expected small-step logistic tail, not missing data. At the deliberately extreme factor, coupled treatment expands through the cellular burned-gas structure, illustrating numerical-method sensitivity rather than an evolved trajectory.*

## Full two-dimensional field galleries

The following unmodified source figures are included at their native resolution so that the weak contours and the cell-wise colorbars remain inspectable. Each uses the retained \(x>0\) domain of an archived H2/O2/Ar cellular-detonation snapshot. In the 12-panel maps, rows show thermochemical state and selector ingredients; the final row compares the original logistic and tuned coupled fractions. These are frozen-state classifications, not new two-dimensional propagation calculations.

### Baseline selector fields at the archived step

![O2 coupled 12-panel two-dimensional field map at the archived step](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-extended-o2-coupled.png)

*Figure — O2-coupled snapshot at \(\Delta t_{\mathrm{code}}=2\times10^{-5}\). The first two rows show temperature, pressure, heat-release magnitude, and the H2/O2/H2O mass fractions. The third row shows chemical and diffusive one-step activities and the shock gate; the last row shows their product and the logistic and tuned coupled fractions. Chemistry and diffusion retain cellular structure, while the very small final coupled fractions show that this short step selects almost pure Strang treatment.* [Open the full-resolution figure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-extended-o2-coupled.png).

![O4 coupled 12-panel two-dimensional field map at the archived step](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-extended-o4-coupled.png)

*Figure — O4-coupled snapshot at the same step. Read the panels in the same order: state variables above, one-step chemical and diffusion activities plus shock gate in the third row, then coupled score and the two output maps below. The wrinkled pressure discontinuity suppresses the shock gate, while the logistic and tuned maps retain only a thin, low-amplitude reaction-layer signature. This permits a like-for-like comparison with the O2 field without treating different archived orders as accuracy rankings.* [Open the full-resolution figure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-extended-o4-coupled.png).

![O4 Strang 12-panel two-dimensional field map at the archived step](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-extended-o4-strang.png)

*Figure — O4-Strang snapshot at the same archived step. The temperature, pressure, composition, activity, and shock-gate panels identify a cellular reaction zone similar to the O4-coupled snapshot. Yet the final coupled-fraction panels remain close to zero because the product score is far below the logistic midpoint at this \(\Delta t\). The image therefore separates visible physical activity from the time-step-dependent request for coupled integration.* [Open the full-resolution figure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-extended-o4-strang.png).

### Fixed-snapshot time-step response

In the next three montages, the left panel is the same unchanged temperature field and the other three panels are the legacy coupled fraction \(1-\chi\) at \(\Delta t/\Delta t_0=10^{-2}\), \(1\), and \(10^3\). The flows do not evolve between panels: only terms that explicitly contain the proposed time step are rescaled.

![O2 coupled snapshot time-step sensitivity montage](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-sensitivity-o2-coupled.png)

*Figure — O2-coupled time-step scan. The first two coupled-fraction panels are nearly dark because the original logistic map stays near its small-step tail. At \(10^3\Delta t_0\), bright contours spread through the cellular burned-gas structure. This broadening is the selector’s response to a much larger numerical update, not a statement that the whole region is a physical reaction zone.* [Open the full-resolution figure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-sensitivity-o2-coupled.png).

![O4 coupled snapshot time-step sensitivity montage](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-sensitivity-o4-coupled.png)

*Figure — O4-coupled time-step scan. Temperature at left is fixed across all columns. The \(10^{-2}\) and base-step fields again have only the small logistic tail; the \(10^3\) panel selects a large part of the resolved cellular structure. Its stronger response than the O2 case follows from the stored local activities, not from a re-run of the detonation.* [Open the full-resolution figure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-sensitivity-o4-coupled.png).

![O4 Strang snapshot time-step sensitivity montage](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-sensitivity-o4-strang.png)

*Figure — O4-Strang time-step scan. The left temperature panel is unchanged, and the three right panels use the same legacy selector at successively larger step factors. The visually quiet base-step panel is quantitatively nonzero but remains far below a strongly coupled threshold; the extreme-step panel demonstrates why a calibration cannot be treated as independent of the temporal discretization.* [Open the full-resolution figure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-sensitivity-o4-strang.png).

![O4 Strang threshold-and-power candidates across time-step factors](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-sensitivity-o4-strang-tuning.png)

*Figure — O4-Strang tuning comparison. Rows are the moderate-transition and maximum-feasible threshold-and-power candidates; columns pair the unchanged temperature field with coupled-fraction maps at \(10^{-2}\), \(1\), and \(10^3\) times the base step. The moderate candidate stays nearly inactive at its calibration step, whereas the maximum-feasible candidate forms a narrow cellular-front band there. Both become much broader at the extreme factor, showing that threshold-and-power tuning shifts activation but does not remove the large-step tendency.* [Open the full-resolution figure](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/2d-sensitivity-o4-strang-tuning.png).

## Early end-to-end results

The first H2/O2 flame comparisons are encouraging on robustness. At coarser time steps, mixed treatments completed cases where a pure Strang calculation with one temporal method lost physical admissibility. In the completed comparisons, the mixed curves generally tracked the fully coupled result more closely than pure Strang.

![Flame-front histories and fitted flame-speed comparison for coupled, split, and mixed treatments](https://raw.githubusercontent.com/harryzhou2000/resources-0/main/2026/reaction-region-indicator-progress/flame_speed_comparison.png)

### Flame speeds

The independently computed reference is \(s_{L,\mathrm{ref}}=2.2531\ \mathrm{m\,s^{-1}}\). Each entry below is the fitted speed in \(\mathrm{m\,s^{-1}}\), followed by its signed relative error in parentheses. `failed` is a numerical failure, not an omitted measurement.

| \(\Delta t\) / treatment | ESDIRK2 | U2R2 | U2R1 |
|---|---:|---:|---:|
| \(7.5\times10^{-4}\) / coupled | 2.699 (+19.773%) | 2.693 (+19.519%) | 2.719 (+20.690%) |
| \(7.5\times10^{-4}\) / Strang | 2.755 (+22.265%) | 2.477 (+9.937%) | 2.739 (+21.553%) |
| \(7.5\times10^{-4}\) / logistic | 2.689 (+19.355%) | 2.712 (+20.373%) | 2.711 (+20.335%) |
| \(7.5\times10^{-4}\) / tuned | 2.696 (+19.650%) | 2.682 (+19.040%) | 2.715 (+20.501%) |
| \(2\times10^{-3}\) / coupled | 2.677 (+18.806%) | 2.672 (+18.574%) | 2.788 (+23.743%) |
| \(2\times10^{-3}\) / Strang | 3.103 (+37.738%) | failed | 2.867 (+27.250%) |
| \(2\times10^{-3}\) / logistic | 2.643 (+17.284%) | 2.628 (+16.624%) | 2.715 (+20.515%) |
| \(2\times10^{-3}\) / tuned | 2.658 (+17.976%) | 2.635 (+16.961%) | 2.722 (+20.829%) |
| \(4\times10^{-3}\) / coupled | 2.627 (+16.597%) | 2.610 (+15.852%) | 2.853 (+26.633%) |
| \(4\times10^{-3}\) / Strang | failed | failed | 3.018 (+33.932%) |
| \(4\times10^{-3}\) / logistic | 2.629 (+16.706%) | 2.590 (+14.953%) | 2.782 (+23.478%) |
| \(4\times10^{-3}\) / tuned | 2.615 (+16.078%) | 2.666 (+18.345%) | 2.756 (+22.341%) |

### Detonation speeds

For detonation, the independent Chapman--Jouguet reference is \(D_{\mathrm{CJ}}=2836.4\ \mathrm{m\,s^{-1}}\). Each entry is the fitted shock speed in \(\mathrm{m\,s^{-1}}\), followed by its signed relative error.

| \(\Delta t\) / treatment | ESDIRK2 | U2R2 | U2R1 |
|---|---:|---:|---:|
| \(2\times10^{-6}\) / coupled | 2791.699 (-1.576%) | 2759.400 (-2.715%) | 2773.981 (-2.201%) |
| \(2\times10^{-6}\) / Strang | 2772.506 (-2.253%) | 2753.164 (-2.935%) | 2770.994 (-2.306%) |
| \(2\times10^{-6}\) / logistic | 2772.768 (-2.243%) | 2753.164 (-2.935%) | 2770.994 (-2.306%) |
| \(2\times10^{-6}\) / tuned | 2772.506 (-2.253%) | 2753.164 (-2.935%) | 2771.050 (-2.304%) |
| \(4\times10^{-6}\) / coupled | 2903.887 (+2.379%) | 2784.194 (-1.841%) | 2793.641 (-1.508%) |
| \(4\times10^{-6}\) / Strang | 2845.226 (+0.311%) | 2842.500 (+0.215%) | 2787.498 (-1.724%) |
| \(4\times10^{-6}\) / logistic | 2844.348 (+0.280%) | 2842.500 (+0.215%) | 2786.752 (-1.750%) |
| \(4\times10^{-6}\) / tuned | 2845.170 (+0.309%) | 2842.500 (+0.215%) | 2787.088 (-1.739%) |
| \(8\times10^{-6}\) / coupled | 3187.864 (+12.391%) | 3721.374 (+31.201%) | 2828.152 (-0.291%) |
| \(8\times10^{-6}\) / Strang | 3212.499 (+13.260%) | 3785.872 (+33.475%) | 2842.297 (+0.208%) |
| \(8\times10^{-6}\) / logistic | 3211.484 (+13.224%) | 3789.052 (+33.587%) | 2850.351 (+0.492%) |
| \(8\times10^{-6}\) / tuned | 3212.025 (+13.243%) | 3787.225 (+33.522%) | 2851.163 (+0.520%) |

There is an important qualification: these are progress results, not a final accuracy claim. The flame calculations are not yet mesh-converged against the independent reference, and nonlinear solves often reached their iteration cap. The evidence is therefore strongest for the selector's localized behavior and its potential to improve robustness; quantitative accuracy still needs better-converged, carefully controlled follow-up calculations.

## Where this is heading

The next step is to test a moderate threshold-and-power variant in posterior flame and detonation calculations. That variant is designed to smooth the edge of the selected region without converting most of a flame to fully coupled integration. The main questions are straightforward: does it preserve the shock-aware localization, does it improve stability at useful time steps, and does it do so without hiding errors that belong to the temporal method or mesh instead?
