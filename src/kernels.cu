// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "kernels.cuh"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <cmath>

namespace swiftware::hpp {

    // GEMM kernel implementation
    __global__ void MM(const float* a, const float* b, float* result, int m, int n, int k)
    {
        // TODO
        // TODO Implement GEMM: C = A * B + C
        // A = m * k
        // B = k * n
        // C = m * n
        int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
        int col = blockIdx.x * blockDim.x + threadIdx.x; // col index

        if (row < m && col < n) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += a[row * k + p] * b[p * n + col];
            }
            result[row * n + col] += sum;  // accumulate
        }
    }

    // GEMV kernel implementation
    __global__ void MV(const float* a, const float* b, float* result, int m, int n)
    {
        // TODO
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        if (i < m) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += a[i * n + j] * b[j];
            }
            result[i] += sum;
        }
    }

    // SpMM kernel implementation
    __global__ void SpMM(int *row_ptr, int *col_id, const float* a, const float* b, float* result, int m, int n, int k)
    {
        // TODO NOT REF  :( 
    }

    // SpMV kernel implementation
    __global__ void SpMV(int *row_ptr, int *col_id, const float* a, const float* b, float* result, int m, int n)
    {
        // TODO NOT REF :( ABANDONDED CHILD 
    }

} // namespace swiftware::hpp
