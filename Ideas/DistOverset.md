---
title: Distance Overset
date: 2025-02-11T14:43:50+08:00
type: post
---

# Distance Overset

距离场隐含几何信息（嵌入）；提高搜索效率；并行

强调（额外）（高阶）：
- 时空精度
- 减少非守恒误差

调研：西工大 蔡晋生 zhengyao ?
    闫超
    二九基地的 谁
    李书杰？not found 

Mesh of background (Cartesian): \(G\)

Mesh (Unstructured) \(A\), \(B\) ...

For mesh \(A\):

- 2 kinds of boundaries: `Wall` and `Far`
- Distance field: \(d_A\)
- Special definitions:
  - Beyond `Wall` boundaries: \(d_A<0\) or \(d_a \equiv-\infty \)
  - Beyond `Far` boundaries: \(d_a\equiv\infty\)
- Interpolation onto $G$:
  - $d_{G,A}$
  - Refinement:
    - In Cartesian cell $\Omega_{G,i}$ of $G$
    - If $\Omega_{G,i}$ has intersection with $\partial \Omega_{A,\text{wall}}$, the $\partial \Omega_{A,\text{wall}}$ mesh is partially stored at $\Omega_{G,i}$, and local $d_{G,A}$ is refined using *more points* or *local bnd representation*
  - Therefore, it can be guaranteed:
    - Given arbitrary point, (to lookup from $G$ or $A$) $d_{G,A}$'s determination of $d_{G,A}\leq0$, $d_{G,A}=\infty$ is identical with $d_A$


For 2 meshes $A$ and $B$:

- For each cell $\Omega_{A,i}$ in $A$:
  - Query points $p \in \Omega_{A,i}$
  - We have $d_{G,A}(p),d_{G,B}(p)$
    - If $\min\{d_{G,A}(p),d_{G,B}(p)\} < 0 \text{ or } \equiv -\infty$, set $p\in\Omega_{Hole}$
    - If $\min\{d_{G,A}(p),d_{G,B}(p)\} > 0$ and $\arg\min\{d_{G,A}(p),d_{G,B}(p)\} = A$, set $p\in\Omega_{A,Field}$
    - If $\min\{d_{G,A}(p),d_{G,B}(p)\} > 0$ and $\arg\min\{d_{G,A}(p),d_{G,B}(p)\} = B$, set $p\in\Omega_{A,Recv,B}$




