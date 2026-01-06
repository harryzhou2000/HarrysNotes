---
title: HPC Notes
date: 2025-01-01 
type: post
slug: hpc-notes
categories: ["Note"]
tags: ["HPC"]
image: ChatGPT_SC_concept.jpg
# https://www.flaticon.com/
---

## DNDSR designs

### Array data structure

#### Array<>

The basic `Array<T,...>` data container is a data container designed for MPI-based distributed parallelism. Element type `T` has to be trivially-copyable.

``` cpp
template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
class Array : public ArrayLayout<T, _row_size, _row_max, _align>
{
    ...
    // holds data (and indexing)
    // methods for indexing
    // methods for serialization
    // methods for device
};
```

|                          | _row_size>=0                      | _row_size==DynamicSize      | _row_size==NonUniformSize |
| ------------------------ | --------------------------------- | --------------------------- | ------------------------- |
| _row_max>=0              | TABLE_StaticFixed                 | TABLE_Fixed                 | TABLE_StaticMax           |
| _row_max==DynamicSize    | TABLE_StaticFixed_row_max ignored | TABLE_Fixed_row_max ignored | TABLE_Max                 |
| _row_max==NonUniformSize | TABLE_StaticFixed_row_max ignored | TABLE_Fixed_row_max ignored | CSR                       |

Underlying data layout: 1-D indexing / 2-D indexing.

For fix-sized 2-D tables, it is basically a row-major matrix with the first index potentially globally aware. For variable-sized 2-D tables, it is a CSR-indexed variable array.

#### ParArray<> and ArrayTransformer<>

Parallel communication: `ParArray<...>` and `ArrayTransformer<...>`.

`ParArray` is `Array` with global indexing info. Generally use a globally (inside the MPI communicator) rank-based contiguous indexing.
Operations on `ParArray` should be collaborative (inside MPI comm).

``` cpp
template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
class ParArray : public Array<T, _row_size, _row_max, _align>
{
    ...
    // holds rank-global indexing data
    // global indexing methods
};
```

`ArrayTransformer<...>` relates two `ParArray<...>` together by defining an all-to-all style arbitrary index mapping.

``` cpp
template <class T, rowsize _row_size = 1, rowsize _row_max = _row_size, rowsize _align = NoAlign>
class ArrayTransformer
{
    ...
    // reference father & son arrays
    // ghost mapping data (father->son index mapping)
    // MPI data types
    // MPI requests
    // pull and push methods
};
```

In PDE solvers, most used critical for ghost mesh/point/cell data, so the mapping is called ghost mapping.

Conceptually:

``` python
ArrSon[IndexSon] = ArrFather[IndexFather]
```

We define that `IndexFather` can be scattered in each rank (section) or even overlapping, and `IndexSon` must be like `0:N`.
Therefore, transforming from `father` to `son` (or `main` to `ghost`) is unique, but not so backwards.

We name `father->son` parallel data transferring `pull`, and `push` when reversed.

General communication procedures:

```cpp
ParArray<...> arrFather(...);
ParArray<...> arrSon(...);
...
// initialize Father's data
ArrayTransformer<...> arrTrans(...);
...
// set arrTrans to refer to arrFather and arrSon
arrFather.createGlobalMapping();
...
// prepare pullingIndex
arrTrans.createGhostMapping(pullingIndex); 
arrTrans.createMPITypes();// automatically reshape arrSon to match arrFather
arrTrans.pullOnce(); // or InitPersistentPull();
```

For persistent data transfers:

```cpp
arrTrans.InitPersistentPull();
// each iteration
{
    arrTrans.StartPersistentPull();
    ...
    arrTrans.WaitPersistentPull();
}
```

To ease the use of underlying data, we have `ArrayEigenMatrix<...>` for AoS style floating point DOF storage, and `ArrayAdjacency<...>` for integer index storage. They inherit from `ParArray<>`.

#### ArrayPair<>

Sometimes a pair of `father` and `son` are always bound together to form a `main` vs. `ghost` pair.

To ease the complexity of data indexing in the ghost region, `ArrayTransformer`'s ghost indexing can be stored as a local version of adjacency table. We define that the local index of `ghost` is appended to the `main` part, so that a local adjacency table can be logically valid and compact. This indexing and pairing is realized by the wrapper `ArrayPair<>`.

```cpp
template <class TArray = ParArray<real, 1>>
struct ArrayPair
{
    // reference father and son arrays
    // hold the ArrayTransformer
    // indexing methods (with appending range)
    // ...
};
```

#### DeviceView

The hierarchy of Array data containers can generate a trivially-copyable device view objects that only hold pointers and sizes and have data-accessors methods that never mutate the array shape.

The device views may hold device pointers, which make them perfectly usable on CUDA devices.

Some shared data, like CSR indexes can be forced into a raw-pointer to reduce overhead (on CPU).

Device views should always be generated and used in a local procedure during which the array shapes are never mutated.

### Asynchronous procedures

A direct design:

``` cpp
ArrayPair a, b;
... // previous computing
a.trans.waitPersistentPull();
computeA(a); // reads a, writes a
a.trans.startPersistentPull();

... // some independent computing

a.trans.waitPersistentPull();
computeA2B(a, b); // reads a, writes b
b.trans.startPersistentPull();

```

A more sophisticated design:

``` cpp
ArrayPair a, b;
... // previous computing
a.trans.startPersistentPull();

computeA2BMain(a, b); // reads a's main part, writes b
a.trans.waitPersistentPull();
computeA2BGhost(a, b); // reads a's main+ghost part, writes b
b.trans.startPersistentPull();
```

A more robust interface: **TASK BASED**

---
---

#### Task based scheduling

##### 1. Core abstraction: tasks, data regions, and access modes

###### 1.1 Data objects with regions

Instead of thinking in terms of arrays + ghost exchange, define **logical data objects** with **regions**:

```cpp
Data a("a"), b("b");

// Regions (can be arbitrary subsets)
Region a_main  = a.region("main");
Region a_ghost = a.region("ghost");
```

The runtime understands that:

* `a_main` is local
* `a_ghost` requires MPI communication

###### 1.2 Tasks with access descriptors

Each task declares **what data it accesses and how**:

```cpp
task ComputeA {
    read_write(a_main);
}

task ComputeA2BMain {
    read(a_main);
    write(b_main);
}

task ComputeA2BGhost {
    read(a_main, a_ghost);
    write(b_main);
}
```

This is the *only* information the scheduler needs.

##### 2. Communication becomes implicit tasks

MPI communication is **not special** — it is just another task:

```cpp
task PullGhost {
    read(a_remote);
    write(a_ghost);
}
```

The runtime inserts this automatically when:

A task reads a region that is not locally available

##### 3. DAG construction (formal model)

Each iteration builds a **directed acyclic graph (DAG)**:

###### Nodes

* Compute tasks
* Communication tasks (MPI Isend/Irecv)
* Optional memory transfer tasks (H2D, D2H, disk I/O)

###### Edges

* `T1 → T2` if T2 needs data produced by T1

##### 4. Scheduler semantics (key rules)

A runtime loop looks like this:

```cpp
while (!all_tasks_done) {
    for task in tasks {
        if (resources_available(task)) {
            launch(task);
        }
    }
    progress_communication();  // MPI_Test, MPI_Waitsome
}
```

###### Task readiness condition

A task is **ready** if:

* All its input regions are available
* No conflicting write is in flight

##### 5. Pseudo-code: user-facing API

```cpp
Graph g;

g.add_task("ComputeA")
  .reads_writes(a_main);

g.add_task("ComputeA2BMain")
  .reads(a_main)
  .writes(b_main);

g.add_task("ComputeA2BGhost")
  .reads(a_main, a_ghost)
  .writes(b_main);

execute(g);
```

##### 6. Runtime-side pseudo-code

###### 6.1 Dependency resolution (for distributed memory)

As non-local dependency should automatically create some comm tasks:

```cpp
for task in graph:
    for region in task.reads:
        if (!region.is_available()):
            insert_comm_task(region)
            add_edge(comm_task, task)
```

###### 6.2 Communication task

```cpp
task CommTask(region) {
    MPI_Irecv(region.buffer);
    MPI_Isend(...);
}
```

Completion of `CommTask` marks `region` as available.

##### 7. Exploiting task parallelism within a rank

Once everything is tasks, **intra-node parallelism is automatic**:

```cpp
#pragma omp parallel
{
    while (scheduler.has_work()) {
        Task t = scheduler.get_ready_task();
        t.run();
    }
}
```

---
---

### Python interface

### DNDSR CUDA kernels

## CUDA

### Sort

### Reduction

### GEMM kernel

### Streams & Graph

A CUDA stream is a **queue** of operations (kernel launches, memcpys, events) that:

* Execute in issue order within the same stream
* May execute concurrently with other streams (if hardware allows)

Default stream

* `0` or `cudaStreamDefault`
* Historically **synchronizes with all other streams** (legacy behavior)
* Modern CUDA supports **per-thread default stream (PTDS)**

```cpp
cudaStream_t s;
cudaStreamCreate(&s);
kernel<<<grid, block, 0, s>>>(args...);
cudaStreamDestroy(s);
```

```cpp
cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s);
cudaMemsetAsync(ptr, value, bytes, s);
```

#### Stream Synchronization

```cpp
cudaStreamSynchronize(s1);
```

* Blocks **host**
* Heavy hammer
* Avoid in performance-critical paths

#### Stream Events

##### Record an event

```cpp
cudaEvent_t e;
cudaEventCreate(&e);
cudaEventRecord(e, s1);
```

##### Make another stream wait

```cpp
cudaStreamWaitEvent(s2, e, 0);
```

##### Result

```
s1:  kernelA → event
s2:                wait → kernelB
```

* No host blocking
* The dependency is like inserted into the queues.

#### Event Sync

```cpp
cudaEventSynchronize(e);
```

Used when correctness requires host visibility.

---

A **CUDA graph** is:

> A **static DAG** of GPU operations captured once and **replayed many times**

* Kernel launch overhead is non-trivial
* Repeated patterns are common (CFD timesteps, solvers)
* Graph replay is **much cheaper** than re-launching kernels

Nodes can be:

* Kernel launches
* Memcpy / memset
* Host functions
* Event record / wait
* Child graphs

#### Graph Construction: Stream capture

```cpp
cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);

kernelA<<<..., s>>>();
kernelB<<<..., s>>>();
cudaMemcpyAsync(..., s);

cudaStreamEndCapture(s, &graph);
```

Then:

```cpp
cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
cudaGraphLaunch(graphExec, s);
```

#### Graph Construction: Manual

```cpp
cudaGraphCreate(&graph, 0);

cudaGraphAddKernelNode(&nodeA, graph, nullptr, 0, &paramsA);
cudaGraphAddKernelNode(&nodeB, graph, &nodeA, 1, &paramsB);
```

### CUDA Performance Analysis

## MPI

### MPI stack

### MPI point-to-point

### MPI communicators

### MPI shared memory

### MPI/CUDA

### MPI optimization ideas

### Others distributed infra

#### RDMA

## CPU HPC

### System and arch

### Multithreading and concurrency

### NUMA

## C++

### Containers

### Multithreading

### Memory management

### Concept and TMP

### Onto CUDA

## CUTLASS, CUB and more

## PyTorch

### Computing graphs and backward()

### Custom operators

### Parallel AI models

## HPC general

### Parallel performance

### Roofline model

### Memory hierarchies

### Performance Analysis
