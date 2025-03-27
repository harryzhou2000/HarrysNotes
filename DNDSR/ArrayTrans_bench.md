---
title: ArrayTransformer bench
date: 2025-03-13T22:31:56+08:00
type: post
---

## What is ArrayTrans

`ArrayTransformer<T,rs,rm,al>` (ArrayTrans) is an operator that handles trans-rank MPI copying of a `ParArray<T,rs,rm,al>`.

ArrayTrans is the foundation of communication during repeated ghost point updating, which is the primary communication pattern of a spacial decomposed parallel PDE solver in DNDS.

## ArrayTrans Bench

Written in Python, the benchmark calls DNDS modules and builds `Array<real, vdim>`, where vdim is a positive size.

[Code is here.](https://github.com/harryzhou2000/DNDSR/blob/33d607b1a0952f5c584f3a5f5f0c416ecd2c0075/script/benchmark/arrayTransBench.py)

To mimic standard workload, use 3-D block (c-indexed) each process.

Each process has size of $32\times32\times32\times\text{vdim}$.

Finds neighbor ranks: upper, lower, front, back, left, right.

Sizes:

- Volume points: $32^3=32768$
- Face points to be pulled: $32^2\times6=6144$, which is \(18.75\%\)

Tested vdim:

- 6
- 20
- 120

> **Pitfall**
> Using mpirun -np X python xxx.py might be dangerous
> If np=64, and OMP_NUM_THREADS (or other thread number control) is not set, and something in python (like NumPy) decides to launch 64 threads, you have 4096 copies of NumPy need to load, which could cause IO system break or memory drain on some systems.
> See this [discussion on stack overflow](https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy).
> On THTJ (thcp1), need to set `export OMP_NUM_THREADS=1` before launch.

## Machines

| Name  | Description | CPU                                                          |
| ----- | ----------- | ------------------------------------------------------------ |
| GS    | gpu704      | Intel(R) Xeon(R) Gold 6326 16-Core x2                        |
| THTJ  | thcp1       | FT 2000 ?                                                    |
| THTJ1 | cp6         | Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz 28-core x2 per node |

## Results

Bandwidth results: bytes/s (per rank / total)

| Name            | $\text{vdim}=6$    | $\text{vdim}=20$    | $\text{vdim}=120$      | Best Total |
| --------------- | ------------------ | ------------------- | ---------------------- | ---------- |
| GS np=16        | 5.8339e+08         | 6.4967e+08          | 6.6650e+08             | 10.2 G     |
| GS np=32        | 2.7550e+08 (0.8ms) | 3.0511e+08 (2ms)    | 2.8825e+08  (17ms)     | 9.3 G      |
| THTJ np=1x56    | 1.5009e+07 (20ms)  | 2.5144e+07 (39ms)   | 2.2339e+07  (264ms)    | 1.3 G      |
| THTJ np=4x56    | 4.2981e+06 (69ms)  | 4.9476e+06 (190ms)  | 5.2787e+06  (1100ms)   | 1.1 G      |
| THTJ np=50x56   | 3.5452e+05 (831ms) | 3.9606e+05 (2481ms) | 4.0971e+05   (14400ms) | 1.1 G      |
| THTJ1 np=1x56   | 1.3897e+08 (2ms)   | 1.2472e+08 (8ms)    | 1.2322e+08 (40 ms)     | 7.4G       |
| THTJ1 np=4x56   | 3.0813e+07 (9ms)   | 2.8067e+07 (35ms)   | 3.5052e+07 (168ms)     | 7.4G       |
| THTJ1 np=20*56  | 8.3952e+06 (35ms)  | 9.3332e+06 (105ms)  | 9.9626e+06 (592ms)     | 10.6G      |
| BSCC-A np=1x64  | 7.5892e+07 (4ms)   | 7.8188e+07 (12ms)   | 8.8352e+07 (67ms)      | 5.4G       |
| BSCC-A np=4x64  | 1.3691e+07         | 1.4866e+07          | 1.7969e+07             | 4.4G       |
| BSCC-A np=20*64 | 2.7962e+06         | 3.0239e+06          | 3.2893e+06             | 4.1G       |

