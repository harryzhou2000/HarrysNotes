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
- [ ] Parallel serializer of Array<>
- [ ] Multilevel assert / exception

## Optimization

- [ ] Better OMP utilization
- [x] OMP implementation in direct methods (Failed, consider using libraries or sub-partitions)
- [ ] Direct methods' structure
- [ ] Comm-Comp Overlapping
  
## Structure improvements

- [ ] Temporary TDof TRec allocator
- [ ] RunImplicitEuler modularize
- [ ] Packed long argument list

## Development

- [ ] Documentation

## Experimental

- [ ] Consecutive 2nd order - to - high order
- [ ] New overset



![alt text](https://harryzhou2000.github.io/resources-0/curtain_A1C-screen.png)