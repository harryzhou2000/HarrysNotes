Great question — this is **exactly the right level** for systems / performance interviews.

I’ll explain **why the (T \times T) score matrix is wasteful**, then **how FlashAttention removes it**, with equations, memory analysis, and a mental model you can reuse in interviews.

---

## 1. Why the (T \times T) score matrix is the problem

Standard attention (single head):

[
S = \frac{QK^\top}{\sqrt{d}}
\quad\in\mathbb{R}^{T\times T}
]

[
A = \text{softmax}(S)
]

[
O = A V
]

### Memory cost

* Scores (S): (O(T^2))
* Attention (A): (O(T^2))

For long sequences:

* (T = 16k \Rightarrow T^2 = 256M) elements
* At FP16 → **512 MB per head** 😬

Even worse:

* These tensors are **materialized in HBM**
* Read again for softmax and matmul with (V)
* Bandwidth dominates compute

👉 **Attention is memory-bound**, not compute-bound.

---

## 2. Key observation behind FlashAttention

We never *need* the full (S) or (A) matrix at once.

For each query position (i):

[
o_i = \sum_{j=1}^{T}
\text{softmax}(s_{ij}) \cdot v_j
]

This depends on:

* one row of (Q)
* all (K, V)
* **but only the final weighted sum**

So the question becomes:

> Can we compute the softmax-weighted sum **without storing all scores**?

**Yes**, using **online softmax** + **tiling**.

---

## 3. Online softmax (core mathematical trick)

Softmax for row (i):

[
\text{softmax}(s_{ij}) = \frac{e^{s_{ij}}}{\sum_k e^{s_{ik}}}
]

This can be computed incrementally.

### Maintain running statistics

For query (i), process keys in blocks:

* Running max: (m)
* Running normalization: (l = \sum e^{s - m})
* Running output accumulator: (o)

#### Initialize

[
m = -\infty,\quad l = 0,\quad o = 0
]

#### For each block of keys (K_b, V_b)

1. Compute scores:
   [
   s_b = q_i K_b^\top
   ]

2. Update max:
   [
   m_{\text{new}} = \max(m, \max(s_b))
   ]

3. Rescale old contributions:
   [
   \alpha = e^{m - m_{\text{new}}}
   ]

4. Update normalization:
   [
   l = l \cdot \alpha + \sum e^{s_b - m_{\text{new}}}
   ]

5. Update output:
   [
   o = o \cdot \alpha + \sum e^{s_b - m_{\text{new}}} \cdot V_b
   ]

6. Set (m = m_{\text{new}})

After all blocks:
[
o_i = \frac{o}{l}
]

✅ **Exact softmax**, no approximation
✅ **No (T \times T) storage**

---

## 4. Tiling: how FlashAttention is actually executed

FlashAttention processes attention in **tiles**:

* Tile queries: (Q_b \in \mathbb{R}^{B_q \times d})
* Tile keys/values: (K_b, V_b \in \mathbb{R}^{B_k \times d})

These tiles are:

* Loaded into **SRAM / shared memory**
* Fully processed
* Discarded immediately

### Kernel structure (simplified)

```text
for each Q-block in sequence:
    load Q_block into SRAM
    initialize m, l, O_block

    for each K/V-block:
        load K_block, V_block into SRAM
        compute scores = Q_block @ K_block^T
        update m, l using online softmax
        update O_block

    write O_block to HBM
```

🚫 No scores written to HBM
🚫 No attention matrix stored
🚫 No softmax matrix stored

---

## 5. Memory & performance comparison

| Method             | Memory   | Bandwidth | Arithmetic Intensity |
| ------------------ | -------- | --------- | -------------------- |
| Standard Attention | (O(T^2)) | Very high | Low                  |
| FlashAttention     | (O(T))   | Minimal   | High                 |

Key result:

> **FlashAttention turns attention from memory-bound to compute-bound**

On GPUs:

* Near Tensor Core peak utilization
* Much better scaling with sequence length

---

## 6. Why this is *exact*, not an approximation

Important interview point:

* FlashAttention computes **exact softmax**
* Uses **algebraic associativity**
* Numerical stability preserved via running max

This is **not**:

* Low-rank approximation
* Sparse attention
* Kernel approximation

It is a **reordering of computation**

---

## 7. Causal masking in FlashAttention

For autoregressive models:

[
s_{ij} = -\infty \quad \text{if } j > i
]

Handled by:

* Skipping masked positions in tiles
* Or explicitly masking scores inside the tile

No extra memory cost.

---

## 8. Backward pass (important!)

FlashAttention backward:

* Recomputes attention on-the-fly
* Stores only:

  * (Q, K, V)
  * (m, l) per row

This is **checkpointing**:

* Trades compute for memory
* Still faster due to bandwidth savings

---

## 9. One-sentence interview summary (very useful)

> *FlashAttention avoids materializing the (T \times T) attention matrix by computing softmax and the value-weighted sum in a single tiled pass using an online softmax formulation, reducing memory from (O(T^2)) to (O(T)) and making attention compute-bound.*

---

If you want next:

* CUDA-level pseudocode (warps / shared memory)
* How FlashAttention maps to **Tensor Cores**
* Differences between FlashAttention v1 / v2 / v3
* How this interacts with **tensor parallelism + NCCL**
* Why PyTorch eager attention is slow

Just tell me.
