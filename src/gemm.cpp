// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemm.h"
#include <immintrin.h>
#include <iostream>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {

    void gemm(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
       // TODO Implement GEMM: C = A * B + C
       // A = m * k
       // B = k * n
       // C = m * n
        // Step 1: Transpose B (k×n → n×k)
        std::vector<float> B_T(n * k);
        for (int row = 0; row < k; ++row) {
            for (int col = 0; col < n; ++col) {
                B_T[col * k + row] = B[row * n + col];
            }
        }

        int tileM = (Sp.TileSize1 > 0) ? Sp.TileSize1 : 32;
        int tileN = (Sp.TileSize2 > 0) ? Sp.TileSize2 : 32;

        // Step 2: Blocked + SIMD GEMM
        #pragma omp parallel for collapse(2)
        for (int ii = 0; ii < m; ii += tileM) {
            for (int jj = 0; jj < n; jj += tileN) {
                int i_end = std::min(ii + tileM, m);
                int j_end = std::min(jj + tileN, n);
                for (int i = ii; i < i_end; ++i) {
                    for (int j = jj; j < j_end; ++j) {
                        __m256 sum_vec = _mm256_setzero_ps();
                        int p = 0;
                        // Now both A[i*k+p..] and B_T[j*k+p..] are contiguous
                        for (; p + 7 < k; p += 8) {
                            __m256 a_vec = _mm256_loadu_ps(&A[i * k + p]);
                            __m256 b_vec = _mm256_loadu_ps(&B_T[j * k + p]);
                            sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
                        }
                        // Horizontal sum
                        float tmp[8];
                        _mm256_storeu_ps(tmp, sum_vec);
                        float sum = tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                                    tmp[4] + tmp[5] + tmp[6] + tmp[7];
                        // Handle leftovers
                        for (; p < k; ++p) {
                            sum += A[i * k + p] * B_T[j * k + p];
                        }
                        C[i * n + j] += sum;
                    }
                }
            }
        }
    }



#ifdef USE_MKL
    void gemmMKL(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
        // MKL GEMM: C = alpha*A*B + beta*C
        // Since C may have initial values, we use beta=1.0 to accumulate
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    m, n, k,
                    1.0f,        // alpha
                    A, k,        // A is m x k
                    B, n,        // B is k x n
                    1.0f,        // beta (accumulate into C)
                    C, n);       // C is m x n

    }
#endif

}