---
title: "Sparser MoE: keeping sparsity end to end"
slug: sparser-moe-end-to-end
date: 2026-09-10T18:00:00+08:00
type: post
categories: ["MoE"]
tags: ["Mixture of Experts", "GPU performance", "distributed training"]
image: cover.png
---

Very large mixture-of-experts models do not necessarily spend most of their time in expert GEMMs. With many experts, a large top-k, high expert parallelism, and a small latent width, the routing and communication machinery can dominate instead: selecting experts, carrying route metadata, moving probabilities, coordinating ranks, and rearranging tokens.

This project treated that as an end-to-end systems problem. The central result is simple: sparsity has to survive the whole path. Selecting only a few experts is not enough if later stages recreate expert-wide metadata, send irrelevant probability columns, or visit inactive local expert slots.

## The workload changes the bottleneck

The motivating GB200 configuration had 2,304 experts, top-36 routing, EP72, 32 local experts per rank, and a 512-wide latent/expert dimension. On average, a token selected only (36/8=4.5) local experts in the representative eight-rank setup. A path that examined all 32 local slots therefore spent about 86% of those checks on inactive experts.

The initial profile reflected that imbalance: router, dispatch, and combine time greatly exceeded grouped expert-GEMM time. The optimization work crossed three layers:

- Transformer Engine: score transformation, top-k selection, and router backward;
- Megatron Core: compact-route eligibility, expansion, padding, recompute, and dispatcher integration; and
- DeepEP HybridEP: metadata scans, dispatch/combine, and expert-order permutations.

An individual kernel improvement mattered only when the compact representation and its correctness invariants made it through all three.

![Conceptual HybridEP workflow: routing metadata is prepared, transported, dispatched to experts, then combined](https://raw.githubusercontent.com/harryzhou2000/megatron-lm-moe-experiments/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/hybrid_ep_workflow.svg)

*Figure — The distributed ownership boundary. Router output originates in Transformer Engine, Megatron Core decides how it may be represented and dispatched, and HybridEP transports, permutes, executes, and combines the resulting token routes. The important engineering constraint was to avoid turning a compact route back into expert-wide work downstream.*

## Preserve a compact route

The original large-expert, large-top-k selection path repeatedly scanned every expert to find one next maximum at a time, with work that grows roughly as `T × E × K`. A radix/threshold selection path replaced repeated full-row scans with bounded histogram passes and a narrowed candidate range. The later router work extended this to forward and backward kernels, using asynchronous loads, persistent grids, lower register pressure, and shape-aware dispatch so small-top-k cases retained their efficient path.

The more consequential representation change was to carry selected expert IDs as a dense list of shape `[T, K]`, rather than an expert-wide boolean map of shape `[T, E]`. At `T=8192`, `E=2304`, and `K=36`, the route metadata changes from 18.87 MB of boolean-map storage to 0.59 MB of int16 selected IDs: a 32x logical reduction.

That replacement is not merely a smaller tensor. Correct integration preserves invalid `-1` routes, expert-tensor-parallel expansion, bias statistics, padding and recompute behavior, backend capability checks, and safe fallback whenever compact routing is not legal. It also keeps two decisions separate: the *route representation* and the *collective used to exchange it*.

## Remove work that has no selected route

Two complementary changes made the sparse representation pay off in HybridEP.

First, a destination received only its local slice of route probabilities. In the EP72 target with 32 local experts, a full FP32 probability row was 9,216 bytes per peer; the local slice was 128 bytes. The controlled probability-carrying dispatch path fell from 185.1 to 103.4 µs, while the no-probability control remained effectively unchanged (95.8 to 95.7 µs). That control ties the gain to removed probability traffic rather than an incidental change.

Second, local pre/post-processing stopped scanning every local expert. A warp ballot identifies selected local slots and iterates only set bits. For the 512-wide latent shape, the permute path fell from 381 to 92 µs and unpermute from 265 to 128 µs. The same idea also supports compact rank and local-expert bitsets for later membership tests.

The combine stage required separate tuning. Dispatch pushes data, whereas combine pulls and reduces it; their latency, queueing, and synchronization constraints are different. Tuning FIFO depth, warp groups, reduction batch, and SM allocation together improved controlled combine-with-probabilities from 960.8 to 252.3 µs (3.8x). The lesson is not a universal parameter recipe—the selected tuple is workload-specific—but that the pipeline must be tuned as a coupled resource budget.

## Matched full-model results

The cleanest comparison held the training recipe fixed and compared a no-tune/no-radix baseline with the complete sparse stack. The final gains were 2.64x–3.30x across the two GPU generations and two large-expert configurations.

| System | Shape | Baseline | Full sparse stack | Gain |
| --- | --- | ---: | ---: | ---: |
| GB200 | 2,304 experts / EP72 | 125.3 | 403.7 TFLOP/s/GPU | 3.22x |
| GB200 | 2,048 experts / EP64 | 159.4 | 421.0 TFLOP/s/GPU | 2.64x |
| GB300 | 2,304 experts / EP72 | 127.2 | 419.7 TFLOP/s/GPU | 3.30x |
| GB300 | 2,048 experts / EP64 | 162.2 | 435.1 TFLOP/s/GPU | 2.68x |

These are matched end-to-end measurements. They should not be confused with the historical progression from roughly 92 to 400 TFLOP/s/GPU, because the surrounding training stack evolved over that period as well.

![Qwen3.5 Sparser 40B EP72 recipe comparison](https://raw.githubusercontent.com/harryzhou2000/megatron-lm-moe-experiments/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/qwen3_5_sparser_40b_ep72.png)

*Figure — Matched median full-model throughput for Qwen3.5 Sparser 40B on GB200, EP72. The optimized stack improves the base recipe by 7.3%, the full-iteration CUDA-graph recipe by 28.4%, and the graph-plus-expert-1F1B recipe by 29.7%. Each pair compares the same recipe before and after the sparse-stack changes.*

![Qwen3 Sparser 80B EP64 recipe comparison](https://raw.githubusercontent.com/harryzhou2000/megatron-lm-moe-experiments/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/qwen3_sparser_80b_ep64.png)

*Figure — Corresponding matched Qwen3 Sparser 80B comparison on GB200, EP64. The gains are larger in this routing-intensive configuration: 53.8% for the base recipe, 33.2% with full-iteration CUDA graphs, and 53.1% with graphs plus expert 1F1B. The chart is a full-model comparison, not an isolated router benchmark.*

## Broader, but different, evidence

A separate MoE-module matrix completed 160 measured iterations for each row with paged stash and full-iteration CUDA graph, without expert 1F1B. It is useful for generalization across shapes, but it is intentionally not interchangeable with the full-model numbers above.

![MoE-module performance across five model shapes](https://raw.githubusercontent.com/harryzhou2000/megatron-lm-moe-experiments/945e8fe0091b315e7328111fff9877cab4eb65df/notes/final_presentation/assets/moe_module_perf_20260719.png)

*Figure — MoE-module microbenchmark results on GB200 across five shapes. The optimized path improves conventional 512-expert proxies by 4.8%–18.1%, while the Qwen3.5 Sparser 40B EP72 shape rises from 234.5 to 484.5 median TFLOP/s/GPU (+106.6%). This figure establishes a shape-dependent trend; it is not full-model training evidence and does not include expert 1F1B.*

## What did not work

Profiling also ruled out attractive-looking shortcuts. Directly writing one remote payload per selected expert increased NVLink traffic and scattered writes; at the target top-k it was projected to be about 4–5x slower than staged dispatch plus local permutation. Fusing dispatch with permute or combine with unpermute added polling and serialization, and was not a robust production win. A more elaborate warp-pruned dense scan also lost to simpler bitsets in several measured shapes.

The broader pattern is useful beyond this workload: removing a kernel is not automatically an optimization. The replacement must also reduce data movement and synchronization on the real critical path.

## What remains

The next questions concern real model-induced skew and overlap: characterize hot experts and rank imbalance across layers and training phases; evaluate dynamic balancing as a complete system including planning and memory movement; and redesign pull/reduce combine around remote-read round trips and barrier cost. These are system-level questions, so they need the same discipline as the results here: distinguish kernel measurements, distributed tests, module benchmarks, and matched full-model evidence.

The underlying [full work summary](https://github.com/harryzhou2000/megatron-lm-moe-experiments/blob/main/notes/sparser_moe_full_work_summary.md) contains detailed implementation, validation, and upstream-integration records.
