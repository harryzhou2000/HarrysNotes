---
title: OMP Implementation in DNDSR
date: 2025-01-21 
type: post
---

Due to the status on THTJ (国家超算中心天津), where the ARM64 section has: 64 cores / node but maximum 56 ranks / node for high speed network, OMP implementation is considered.

## Basic Idea

Use macro `DNDS_DIST_MT_USE_OMP` to enable OMP related `#pragma`-s. Basically it influences the distributed computation part, making DOF/REC vector operations, SGS operations, RHS evaluation, reconstruction, limiting parallelized (in threads) with OMP.

In runtime, `DNDS_DIST_OMP_NUM_THREADS` environment variable controls the omp_get_max_threads() in the distributed computation routines.

## Implementation

See `34abb36ca9418c123a7d516b6b68a6fea80ea8bc`.

## Running

To make OMP nested in MPI (MPI+OMP) work, core binding needs to be tended. See this [document](https://lab.cs.tsinghua.edu.cn/hpc/doc/faq/binding/) for guides.

For running on `gpu704` + `OpenMPI` with arbitrary `np` and `DNDS_DIST_MT_USE_OMP` it is tested that this works:

``` bash
mpirun --bind-to none -np 8 app/eulerSA.exe ../cases/eulerSA_config_0012_AOA15.json
```

## Performance issues

All tests carried out on `gpu704`.

Set DNDS_UNSAFE_MATH_OPT to On and others default.

### Recovery

Reference case: `cases/eulerSA_config_0012_AOA15.json` see Appendix.

When `DNDS_DIST_MT_USE_OMP` is not defined, we hope it does not affect performance.

Before OMP implementation: use `4743a171299e0403f2230780db1f418ea0815b70`, 10 iter time average: 
3.035.

After (`34abb36ca9418c123a7d516b6b68a6fea80ea8bc`): 
3.025

No change (because the macro handles statically...).

If `DNDS_DIST_MT_USE_OMP` is defined but `DNDS_DIST_OMP_NUM_THREADS` is not set or set to `1`, on `34abb36ca9418c123a7d516b6b68a6fea80ea8bc`:
3.113

That is \(2.9\%\)  slower. Due to the need to add a face-flux buffer in the EvaluateRHS.

Other optimizations & summary:

| type                                         | time (s/10 iter)   |
| -------------------------------------------- | ------------------ |
| Before OMP implementation                    | 3.035              |
| After OMP implementation (OMP_impl)          | 3.025 (\(-0.3\%\)) |
| OMP_impl + `DNDS_DIST_MT_USE_OMP`            | 3.113 (\(+2.6\%\)) |
| OMP_impl + UDOF_Pool                         | 3.049 (\(+0.5\%\)) |
| OMP_impl + UDOF_Pool + `DNDS_NDEBUG`         | 2.850 (\(-6.1\%\)) |
| OMP_impl + UDOF_Pool + jemalloc/tcmalloc_min | 2.995 (\(-1.3\%\)) |


### Acceleration

#### Test 1

Note that `OMP_SCHEDULE` should be set to improve scheduling.

``` bash
DNDS_DIST_OMP_NUM_THREADS=2 OMP_SCHEDULE=STATIC  mpirun --bind-to none -np 4 app/eulerSA.exe ../cases/eulerSA_config_0012_AOA15.json
```

Result:
3.986

- At `ce71614e8651c2c738e8a31304f9b46fb05cc0f1`:
    - 3.533: 2x4
    - 3.149: 1x8

More than \( 30\% \) slowdown.

#### Test 2

``` bash
DNDS_DIST_OMP_NUM_THREADS=1 OMP_SCHEDULE=GUIDED  mpirun --bind-to none -np 32 app/eulerSA.exe ../cases/eulerSA_config_0012_AOA15.json
```

Result:
1.029

``` bash
DNDS_DIST_OMP_NUM_THREADS=2 OMP_SCHEDULE=GUIDED  mpirun --bind-to none -np 16 app/eulerSA.exe ../cases/eulerSA_config_0012_AOA15.json
```

Result:
1.168
Slow-down = \( 14\% \)

<!-- 
\[
\% \pdv{x}{u}
\] -->

## Appendix

Record for `cases/eulerSA_config_0012_AOA
15.json`.

```json
{
    "timeMarchControl": {
        "dtImplicit": 1e100,
        "nTimeStep": 1,
        "steadyQuit": true,
        "useRestart": false,
        "odeCode": 0,
        "tEnd": 3e+200
    },
    "convergenceControl": {
        "nTimeStepInternal": 10000,
        "rhsThresholdInternal": 1e-8,
        "res_base": 0.0,
        "useVolWiseResidual": true
    },
    "outputControl": {
        "nConsoleCheck": 1,
        "nConsoleCheckInternal": 10,
        "consoleOutputMode": 1,
        "nDataOutC": 100,
        "nDataOut": 10,
        "nDataOutCInternal": 50,
        "nDataOutInternal": 1000000000,
        "nRestartOut": 100,
        "nRestartOutC": 10,
        "nRestartOutCInternal": 200,
        "nRestartOutInternal": 1000000000,
        "tDataOut": 3e+200,
        "consoleMainOutputFormatInternal": [
            "\t Internal === Step [{step:4d},{iStep:2d},{iter:4d}]   ",
            "res {termRed}{resRel:.15e}{termReset}   ",
            "t,dT,dTaumin,CFL,nFix {termGreen}[{tSimu:.3e},{curDtImplicit:.3e},{curDtMin:.3e},{CFLNow:.3e},[alphaInc({nLimInc},{alphaMinInc:.3g}), betaRec({nLimBeta},{minBeta:.3g}), alphaRes({nLimAlpha},{minAlpha:.3g})]]{termReset}   ",
            "Time[{telapsedM:.3f}] recTime[{trecM:.3f}] rhsTime[{trhsM:.3f}] commTime[{tcommM:.3f}] limTime[{tLimM:.3f}] limTimeA[{tLimiterA:.3f}] limTimeB[{tLimiterB:.3f}]"
        ],
        "dataOutAtInit": true
    },
    "implicitCFLControl": {
        "CFL": 50,
        "nForceLocalStartStep": 2147483647,
        "nCFLRampStart": 2147483647,
        "nCFLRampLength": 2147483647,
        "CFLRampEnd": 0.0,
        "useLocalDt": true
    },
    "dataIOControl": {
        "uniqueStamps": false,
        "meshRotZ": 0.0,
        "meshFile": "../data/mesh/NACA0012_H2.cgns",
        "outPltName": "../data/out/NACA0012_H2-MGtest",
        "outPltMode": 0,
        "readMeshMode": 0,
        "outPltTecplotFormat": true,
        "outPltVTKFormat": false,
        "outAtPointData": true,
        "outAtCellData": true,
        "outBndData": true,
        "outCellScalarNames": [
            // "minJacobiDetRel",
            // "cellVolume",
            "dWall"
        ]
    },
    "boundaryDefinition": {},
    "implicitReconstructionControl": {
        "nInternalRecStep": 1,
        "zeroGrads": false,
        "recLinearScheme": 0, // 2 is fpcg, 1 is gmres
        "nGmresSpace": 5,
        "nGmresIter": 1,
        "fpcgResetScheme": 0,
        "fpcgResetReport": 1,
        "recThreshold": 1e-05,
        "nRecConsolCheck": 10,
        "storeRecInc": true,
        "dampRecIncDTau": false
    },
    "limiterControl": {
        "useLimiter": false,
        "smoothIndicatorProcedure": 0,
        "limiterProcedure": 1, //1 == CWBAP
        "nPartialLimiterStart": 1000000,
        "nPartialLimiterStartLocal": 1000000
    },
    "vfvSettings": {
        "maxOrder": 3,
        "intOrder": 5,
        "intOrderVR": 5,
        "cacheDiffBase": true,
        "jacobiRelax": 1.0,
        "SORInstead": false,
        "smoothThreshold": 1e-3,
        "WBAP_nStd": 10.0,
        "normWBAP": false,
        "subs2ndOrder": 1,
        "subs2ndOrderGGScheme": 0,
        "baseSettings": {
            "localOrientation": true,
            "anisotropicLengths": false
        },
        "functionalSettings": {
            // "scaleType": "MeanAACBB",
            "dirWeightScheme": "HQM_OPT",
            // "dirWeightScheme": "ManualDirWeight",
            // "manualDirWeights": [
            //     1.0,
            //     1,
            //     0,
            //     0
            // ],
            "geomWeightScheme": "HQM_SD",
            "geomWeightPower": 0.5,
            "geomWeightBias": 1,
            // "geomWeightScheme": "SD_Power",
            // "geomWeightPower1": -0.5,
            // "geomWeightPower2": 0.5,
            // "useAnisotropicFunctional": true,
            // // "anisotropicType": "InertiaCoordBB",
            // "inertiaWeightPower": 0,
            // "scaleMultiplier": 1,
            "_tail": 0
        }
    },
    "linearSolverControl": {
        "jacobiCode": 0,
        "sgsIter": 0,
        "sgsWithRec": 0,
        "gmresCode": 1,
        "gmresScale": 2,
        "nGmresSpace": 5,
        "nGmresIter": 2,
        "multiGridLP": 1,
        "multiGridLPInnerNIter": 4,
        "directPrecControl": {
            "useDirectPrec": true,
            "iluCode": 2,
            "orderingCode": 4,
            "_tail": 0
        }
    },
    "others": {
        "nFreezePassiveInner": 0
    },
    "eulerSettings": {
        "ignoreSourceTerm": false,
        "useScalarJacobian": false,
        "useRoeJacobian": false,
        "riemannSolverType": "Roe_M1",
        "uRecBetaCompressPower": 1,
        "wallDistScheme": 1,
        "wallDistExection": 5,
        "wallDistRefineMax": 0.01,
        "wallDistIter": 1000,
        "wallDistLinSolver": 0,
        "wallDistResTol": 1e-4,
        "wallDistPoissonP": 8,
        "ransUseQCR": 0,
        "ransSARotCorrection": 1,
        "usePrimGradInVisFlux": 0,
        // "ransForce2nd": 1,
        "constMassForce": [
            0.0,
            0.0,
            0.0
        ],
        "Comment": {
            "M0.15": [
                1,
                1,
                0,
                0,
                79.8650793650794,
                1.3,
                {
                    "Rgas": 0.116221972344982
                }
            ],
            "M0.8": [
                1,
                1,
                0,
                0,
                2.971576866040534,
                1.3,
                {
                    "Rgas": 0.003338277043445
                }
            ]
        },
        "farFieldStaticValue": [
            1,
            1,
            0,
            0,
            79.8650793650794,
            3
        ],
        // "farFieldStaticValue": [
        //     10000,
        //     96.5925826289068,
        //     25.8819045102521,
        //     0,
        //     79.8650793650794,
        //     1.3
        // ],
        "idealGasProperty": {
            "gamma": 1.4,
            "Rgas": 0.116221972344982,
            // "Rgas": 0.0000116221972344982,
            "muGas": 3.472222222222222e-7,
            // "muGas": 3.472222222222222e-5,
            "prGas": 0.72,
            "TRef": 273.15,
            "CSutherland": 110.4
        },
        "cLDriverSettings": {
            "AOAInit": 5.0,
            "AOAAxis": "z",
            "CL0Axis": "y",
            "CD0Axis": "x",
            "refArea": 1.0,
            "refDynamicPressure": 0.5,
            "targetCL": 1.0,
            // "nIterStartDrive": 1000,
            "nIterConvergeMin": 100,
            "CLconvergeThreshold": 0.001,
            "CLconvergeWindow": 10
        },
        "cLDriverBCNames": [
            "WALL"
        ]
    },
    "restartState": {
        "iStep": 1,
        "iStepInternal": 15000,
        "odeCodePrev": -1,
        "lastRestartFile": "../data/out/NACA0012_H2__C_p16_restart_test1.dir"
    }
}

```