---
title: Multigrid Test
date: 2025-02-27T16:54:08+08:00
type: post
---

# Multigrid Test

## MG-FAS

### 非线性问题

- 非线性问题：

$$
𝐹(𝑥)=0
$$

- 牛顿迭代：

$$
\pdv{F}{x}\Delta x = -F(x)
$$

- 近似+松弛的牛顿迭代：

$$
\left(\widehat{\pdv{F}{x}} - \tau^{-1}\right)\Delta x = -F(x)
$$

- 松弛迭代（非线性）：
  - 逐点更新$𝑥_𝑖$（可同时更新$𝐹_𝑖$，如非线性GS）
  - 牛顿迭代可以视作一种特殊的松弛
  - 牛顿迭代内部采用不同的线性求解器，也可以是松弛迭代

FAS中，计算\(∆𝑥\)更新求解本级的\(𝐹(𝑥)=0\)，即为一个松弛步/光滑步。

### FAS 通用过程

- Relax：通过密网格残差 $𝐹^ℎ (𝑥^{ℎ,𝑚})$ 计算  $𝑦^{ℎ,𝑚}=𝑥^{ℎ,𝑚}+∆𝑦^{ℎ,𝑚}$
- Restrict：$𝑦^{2ℎ,𝑚}=𝐼_ℎ^{2ℎ} 𝑦^{ℎ,𝑚}$,  $𝑟^{2ℎ}=𝐼_ℎ^{2ℎ} 𝐹^ℎ (𝑦^{ℎ,𝑚})$
- Relax： 2h网格上的待解方程：$𝐹^{2ℎ} (𝑥^{2ℎ} )=𝐹^{2ℎ} (𝑦^{2ℎ,𝑚} )−𝑟^{2ℎ}$
  - 其中初值即为$𝑥^{2ℎ,0}=𝑦^{2ℎ,𝑚}$
  - 求解后得到$𝑥^{2ℎ,𝑙𝑎𝑠𝑡}=𝑥^{2ℎ,0}+𝑒^{2ℎ, 𝑚}$
- Interpolate: $𝑒^{ℎ, 𝑚}=𝐼_{2ℎ}^ℎ 𝑒^{2ℎ, 𝑚}$, $𝑥^{ℎ,𝑚+1}=𝑦^{ℎ,𝑚}+𝑒^{ℎ, 𝑚}$

此处的$F^{2h}$为疏网格的算子。

投影/限制算子：$𝐼_ℎ^{2ℎ}$ 与 插值算子： $𝐼_{2ℎ}^ℎ$ 为线性算子。

### 针对双时间步/伪时间步的说明：

$$
F(x)=\alpha R(x) - \frac{x}{\Delta t} + B
$$

其中 $R(x)=\dv{x}{t}, \alpha > 0$

两级网格间的操作：

- Relax：通过密网格残差 
    $$
    \widehat{𝐹}^ℎ (𝑥^{ℎ,𝑚}) = \alpha R^{h}(x^{ℎ,𝑚}) - \frac{x^{ℎ,𝑚}}{\Delta t} + B^{h}
    $$
   计算  $𝑦^{ℎ,𝑚}=𝑥^{ℎ,𝑚}+∆𝑦^{ℎ,𝑚}$
- Restrict：$𝑦^{2ℎ,𝑚}=𝐼_ℎ^{2ℎ} 𝑦^{ℎ,𝑚}$
    $$
        B^{2h}=𝐼_ℎ^{2ℎ}\alpha R^h(𝑦^{ℎ,𝑚}) + 𝐼_ℎ^{2ℎ} B^h - \alpha R^{2h}(𝑦^{2ℎ,𝑚})
    $$
- Relax： 2h网格上的待解方程：
    $$
    \widehat{𝐹}^{2ℎ} (𝑥^{2ℎ}) = \alpha R^{2h}(𝑥^{2ℎ}) - \frac{𝑥^{2ℎ}}{\Delta t} + B^{2h}
    $$
  - 其中初值即为$𝑥^{2ℎ,0}=𝑦^{2ℎ,𝑚}$
  - 求解后得到$𝑥^{2ℎ,𝑙𝑎𝑠𝑡}=𝑥^{2ℎ,0}+𝑒^{2ℎ, 𝑚}$
  - Remark: 若 $\widehat{𝐹}^ℎ (y^{ℎ,𝑚})=0$，则有$\widehat{𝐹}^{2ℎ} (y^{2ℎ,𝑚})=0$
- Interpolate: $𝑒^{ℎ, 𝑚}=𝐼_{2ℎ}^ℎ 𝑒^{2ℎ, 𝑚}$, $𝑥^{ℎ,𝑚+1}=𝑦^{ℎ,𝑚}+𝑒^{ℎ, 𝑚}$

此处的$\hat{F}^{h},\hat{F}^{2h}$为两层网格各自的残差方程，注意并非是最密网格上残差算子投影到疏网格上。

### 多级 FAS

以下描述1次最密网格解更新的过程

示例：不做MG

- Relax on $h$

示例：1层MG

- Relax on $h$
  - Restrict: $h\rightarrow 2h$
  - Relax on $2h$ for $n_1$ times
  - Interpolate: $2h\rightarrow h$

示例：2层MG: V cycle

- Relax on $h$
  - Restrict: $h\rightarrow 2h$
  - Relax on $2h$ for $n_1$ times
    - Restrict: $2h\rightarrow 4h$
    - Relax on $4h$ for $n_2$ times
    - Interpolate: $4h\rightarrow 2h$
  - Relax on $2h$ for $m_1$ times
  - Interpolate: $2h\rightarrow h$


示例：2层MG: W cycle

- Relax on $h$
  - Restrict: $h\rightarrow 2h$
  - Relax on $2h$ for $n_1$ times
    - Restrict: $2h\rightarrow 4h$
    - Relax on $4h$ for $n_2$ times
    - Interpolate: $4h\rightarrow 2h$
  - Relax on $2h$ for $m_1$ times
    - Restrict: $2h\rightarrow 4h$
    - Relax on $4h$ for $n_2$ times
    - Interpolate: $4h\rightarrow 2h$
  - Relax on $2h$ for $k_1$ times
  - Interpolate: $2h\rightarrow h$







