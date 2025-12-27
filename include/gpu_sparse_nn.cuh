// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB01_GPU_SPARSE_NN_CUH
#define LAB01_GPU_SPARSE_NN_CUH

#include "def.h"

namespace swiftware::hpp {

// GPU sparse NN forward pass (batchSize = 1 only)
DenseMatrix *sparseNNGPU(DenseMatrix *InData,CSR *W1, CSR *W2,DenseMatrix *B1, DenseMatrix *B2);

// SPMV kernel
__global__ void spmvKernel(
    const int *Ap,
    const int *Ai,
    const float *Ax,
    const float *x,
    float *out,
    const float *bias,
    int m);

// SPMM kernel
__global__ void spmmKernel(
    const int *Ap,
    const int *Ai,
    const float *Ax,
    const float *B,
    float *C,
    int m, int n, int k);

// (optional) relu kernel if needed outside
__global__ void relu_kernel(float *data, int size);

// (optional) argmax
__global__ void argmax_kernel_sparse(const float* z, int n, int* pred);

}

#endif //LAB01_GPU_SPARSE_NN_CUH
