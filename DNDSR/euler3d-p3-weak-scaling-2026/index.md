---
title: "DNDSR Euler3D P3 weak scaling on four CPU clusters"
slug: dndsr-euler3d-p3-weak-scaling
date: 2026-10-19T11:18:05+08:00
type: post
draft: false
categories: ["DNDSR"]
tags: ["CFD", "HPC", "MPI", "benchmark"]
image: weak-scaling.png
---

This study measures the pure-MPI weak scaling of DNDSR's fourth-order
Euler3D solver on four CPU clusters. Every run uses complete nodes, one MPI
rank per physical core, and about 1,024 periodic hexahedral cells per rank.
The measured kernel is the steady internal update of the P3 VFV scheme
(`maxOrder=3`, quadrature `intOrder=5`, limiter disabled).

Each accepted performance record has 500 solver updates. The first 100 are
warmup, and the statistics use all 400 remaining timings without trimming.
The **mean** includes every stall; the **median** represents the typical
update. This distinction matters at high rank counts, where occasional long
updates can materially reduce sustained throughput.

The complete audit has **31 runs with all 400 measured updates**:

- 22 production weak-scaling runs using serial CGNS input;
- five workload-matched distributed-HDF5/new-MPI retests;
- four distributed-HDF5 diagnostic controls with deliberately heavier
  per-rank workloads.

Runs that stopped during mesh setup have no solver-update timings and are not
reported as performance points.

## Machines and MPI stacks

| Cluster | CPU per node | Physical cores/node | Benchmark partition | Production MPI | Supplemental MPI |
|---|---|---:|---|---|---|
| m8 | 2x AMD EPYC 9654, Zen 4 | 192 | `amd_m8_768` | Open MPI 4.1.0, UCX 1.14 | Open MPI 5.0.10, UCX 1.14 |
| m9 | 2x AMD EPYC 9755, Zen 5 | 256 | `amd_m9_768` | Open MPI 4.1.8, UCX 1.18 | same stack |
| a8 | 2x AMD EPYC 9554, Zen 4 | 128 | `amd_a8_384` | Open MPI 4.1.0, UCX 1.14 | Open MPI 4.1.6, UCX 1.14 |
| thtj1 | 2x Intel Xeon Gold 6348, Ice Lake-SP | 56 | `cp6` | Open MPI 4.1.4, site UCX/GLEX | none |

The a8 runs used the normal 384 GB class rather than the dedicated 768 GB
large-memory partition. The m9 runs likewise used the normal 768 GB partition,
not its 1.5 TB partition. m8 used its M8-specific 768 GB partition, and thtj1
used the normal `cp6` production partition.

## Production weak scaling

![Mean throughput and weak-scaling efficiency](weak-scaling.png)

[Download the vector PDF](weak-scaling.pdf).

The horizontal axis is the number of physical CPU cores. The left panel shows
sustained mean throughput; the right panel normalizes it by the corresponding
one-node result.

| Cluster | Nodes | Ranks | Cells/rank | Mean MCell-it/s | Median MCell-it/s | Mean efficiency | Median efficiency | Timing CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| m8 | 1 | 192 | 1016.2 | 1.772 | 1.830 | 100.0% | 100.0% | 0.108 |
| m8 | 2 | 384 | 1013.1 | 3.504 | 3.672 | 98.9% | 100.3% | 0.150 |
| m8 | 4 | 768 | 1013.9 | 6.261 | 6.527 | 88.3% | 89.2% | 0.161 |
| m8 | 8 | 1536 | 1016.2 | 9.366 | 12.569 | 66.1% | 85.8% | 2.941 |
| m8 | 16 | 3072 | 1034.0 | 20.358 | 23.774 | 71.8% | 81.2% | 0.307 |
| m9 | 1 | 256 | 1024.0 | 2.382 | 2.808 | 100.0% | 100.0% | 1.599 |
| m9 | 2 | 512 | 1038.0 | 5.103 | 5.450 | 107.1% | 97.1% | 0.371 |
| m9 | 4 | 1024 | 1036.3 | 9.238 | 10.545 | 96.9% | 93.9% | 0.657 |
| m9 | 8 | 2048 | 1024.0 | 15.264 | 20.323 | 80.1% | 90.5% | 0.773 |
| m9 | 16 | 4096 | 1018.9 | 21.936 | 34.915 | 57.5% | 77.7% | 1.077 |
| m9 | 32 | 8192 | 1021.2 | 30.201 | 44.441 | 39.6% | 49.5% | 0.978 |
| a8 | 1 | 128 | 1036.3 | 1.656 | 1.712 | 100.0% | 100.0% | 0.082 |
| a8 | 4 | 512 | 1038.0 | 5.249 | 6.177 | 79.3% | 90.2% | 2.092 |
| a8 | 8 | 1024 | 1036.3 | 11.211 | 11.790 | 84.6% | 86.1% | 0.171 |
| a8 | 16 | 2048 | 1024.0 | 19.793 | 21.288 | 74.7% | 77.7% | 0.173 |
| thtj1 | 1 | 56 | 1059.3 | 0.525 | 0.528 | 100.0% | 100.0% | 0.029 |
| thtj1 | 2 | 112 | 1050.4 | 1.030 | 1.040 | 98.0% | 98.4% | 0.042 |
| thtj1 | 4 | 224 | 1013.3 | 2.036 | 2.076 | 96.9% | 98.2% | 0.060 |
| thtj1 | 8 | 448 | 1019.0 | 3.980 | 4.109 | 94.7% | 97.2% | 0.103 |
| thtj1 | 16 | 896 | 1018.6 | 7.309 | 7.639 | 87.0% | 90.4% | 0.119 |
| thtj1 | 32 | 1792 | 1013.3 | 2.101 | 2.109 | 12.5% | 12.5% | 0.011 |
| thtj1 | 64 | 3584 | 1019.0 | 4.414 | 4.241 | 13.1% | 12.5% | 0.177 |

m9 reaches the largest accepted production point: 8,192 ranks on 32 nodes,
with 30.20 million cell-iterations/s sustained mean throughput and 44.44
million median throughput. The mean and median efficiencies are 39.6% and
49.5%. Their separation, together with CV 0.978, shows that the typical update
is considerably faster than the stall-inclusive average.

m8 and a8 retain 71.8% and 74.7% mean efficiency at 16 nodes. thtj1 scales
smoothly through 16 nodes, but throughput falls from 7.31 million
cell-iterations/s at 16 nodes to 2.10 million at 32 nodes. Its 32-node CV is
only 0.011, so this is a stable scaling cliff rather than a few outlier
updates. The 64-node run recovers some global throughput but remains near 13%
efficiency. Its attempted 128-node runs never reached solver updates and thus
do not define a throughput point.

## Per-core throughput and MPI retests

![Mean and median throughput per physical CPU core](per-core-throughput.png)

[Download the vector PDF](per-core-throughput.pdf).

The five workload-matched HDF5 retests stay within 1.2% of the 1,024-cell
target. They test the large-rank path with validated MPI stacks available on
the same machines.

| Cluster | Nodes | Ranks | Cells/rank | MPI stack | Mean MCell-it/s | Median MCell-it/s | Mean kCell-it/s/core | Median kCell-it/s/core | CV | Outcome |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| a8 | 1 | 128 | 1036.3 | Open MPI 4.1.6 + UCX 1.14 | 1.611 | 1.679 | 12.584 | 13.116 | 0.153 | completed |
| a8 | 32 | 4096 | 1018.9 | Open MPI 4.1.6 + UCX 1.14 | 29.751 | 33.806 | 7.263 | 8.253 | 0.561 | completed |
| m8 | 1 | 192 | 1016.2 | Open MPI 5.0.10 + UCX 1.14 | 1.792 | 1.954 | 9.335 | 10.177 | 0.299 | completed |
| m8 | 32 | 6144 | 1030.5 | Open MPI 5.0.10 + UCX 1.14 | 12.421 | 13.758 | 2.022 | 2.239 | 0.944 | completed |
| m9 | 16 | 4096 | 1018.9 | Open MPI 4.1.8 + UCX 1.18 | 21.218 | 27.865 | 5.180 | 6.803 | 0.977 | solver completed; allocation cleanup timed out |

The newer stacks make the prior failure-scale workloads complete on both m8
and a8. This is a direct A/B result for completion behavior: the executable
and UCX version are retained while the Open MPI stack changes. HDF5 input by
itself did not make the old Open MPI 4.1.0 large-rank attempts complete.

Performance still differs sharply. Relative to the matching one-node retest,
a8 retains 57.7% mean and 62.9% median efficiency at 32 nodes. m8 retains only
21.7% mean and 22.0% median efficiency, with CV 0.944. The newer m8 stack fixes
the completion problem but does not fix its large-scale performance.

The m9 HDF5 result produced all 400 samples, and the application and MPI steps
exited successfully. Only the outer allocation reached its shortened time
limit during cleanup, so the performance sample is accepted with that status
shown explicitly.

## Heavier diagnostic controls

The remaining four complete records use 1.33 to 15.92 times the target
cells per rank. They are valid performance measurements for those workloads,
but they are not weak-scaling members and are excluded from both comparison
curves.

| Cluster | Nodes | Ranks | Cells/rank | Relative load | Mean MCell-it/s | Median MCell-it/s | Mean kCell-it/s/core | Median kCell-it/s/core | CV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a8 | 16 | 2048 | 2037.7 | 1.99x | 20.049 | 21.133 | 9.789 | 10.319 | 0.117 |
| a8 | 24 | 3072 | 1358.5 | 1.33x | 23.292 | 25.871 | 7.582 | 8.422 | 0.599 |
| m8 | 16 | 3072 | 2061.1 | 2.01x | 21.520 | 24.011 | 7.005 | 7.816 | 0.196 |
| m9 | 1 | 256 | 16301.9 | 15.92x | 2.716 | 2.785 | 10.608 | 10.879 | 0.160 |

## Interpretation and scope

The benchmark demonstrates three separate limits:

1. m9 can execute the unchanged production path beyond 5,000 ranks, reaching
   8,192 ranks, so there is no universal rank-count ceiling near that value.
2. The Open MPI 4.1.0 production stacks on m8 and a8 are strongly associated
   with their repeatable large-rank setup failures. Validated newer stacks
   complete the same scale, but m8 still shows poor sustained efficiency.
3. thtj1's large-node limit is different: its 32-node solver updates are
   consistently slow, while its 128-node attempts spend hours in distributed
   mesh setup and never produce update timings.

This is a full-node, pure-MPI application benchmark, not a controlled CPU
microarchitecture comparison. Core counts, processor models, MPI versions,
UCX versions, and fabrics differ among clusters. Mean/median gaps and timing
CVs should be read alongside throughput before drawing conclusions from a
single high-rank point.

The solver source is DNDSR `dev/harry` commit
`f6ece7e91e97c6337ef0761c30017b2ddad6b24b`. Exact configuration hashes,
machine-readable timing rows, scheduler classifications, and failure evidence
are in the public
[benchmark report](https://github.com/CFDLAB-THU/DNDSR-cluster-helper/blob/main/benchmark/REPORT.md).
