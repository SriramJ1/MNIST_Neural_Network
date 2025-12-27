// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#ifndef LAB01_GPU_DENSE_NN_CUH
#define LAB01_GPU_DENSE_NN_CUH
#pragma once
#include "gpu_utils.h"
#include <cuda_runtime.h>

namespace swiftware::hpp
{
    DenseMatrix* dense_nn_gemm_gpu(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2,
                                   DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);

    DenseMatrix* dense_nn_gemv(DenseMatrix* InData, DenseMatrix* W1, DenseMatrix* W2, DenseMatrix* B1, DenseMatrix* B2, ScheduleParams Sp);
    __global__ void bias_add_kernel(float* H, const float* B, int M, int N);
    __global__ void tanh_kernel(float* data, int size);
    __global__ void sigmoid_kernel(float* data, int size);
    __global__ void argmax_kernel(const float* H, int M, int N, int* pred);

    __global__ void transpose_kernel(const float* in, float* out, int rows, int cols);





}
#endif //LAB01_GPU_DENSE_NN_CUH