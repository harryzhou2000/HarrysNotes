---
title: Plans for DNDSR
date: 2025-01-21 
type: post
# image: default-cover.png
---

# Plans for DNDSR

## New features

- [ ] 2nd order standalone FV solver
- [ ] Modal DG solver
- [ ] multi-block support
- [ ] Cartesian interpolator + FFT
- [ ] Point sampler
- [x] Parallel serializer of Array<>: HDF5 now usable, but different partition not inter-operable
- [ ] Array / Solution / Mesh partition merger / remapping
- [ ] Multilevel assert / exception
- [x] Reordering local cells
- [x] Write mesh with CGNS (parallel): need 4.5.0

## Optimization

- [x] Better OMP utilization: most places OMPed
- [x] OMP implementation in direct methods (Failed, consider using libraries or sub-partitions)
- [ ] Sub-partition for direct methods in OMP
- [ ] Direct methods' code structure
- [ ] Comm-Comp Overlapping
  
## Structure improvements

- [x] Temporary TDof TRec allocator
- [ ] RunImplicitEuler modularize: in proceeding
- [ ] Packed long argument list
  
## Build System

- [x] CMake distinguishes EulerModel's
- [x] CCache acceleration
- [ ] CMake use external project to organize cfd_external?

## Development

- [ ] Documentation

## Experimental

- [ ] Consecutive 2nd order - to - high order
- [ ] Implicit residual smoothing
- [ ] New overset
- [ ] Exponential Time Marching
- [ ] FV P-Multigrid

![alt text](https://harryzhou2000.github.io/resources-0/curtain_A1C-screen.png)