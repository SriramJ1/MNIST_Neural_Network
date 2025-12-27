// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemv.h"
#include <immintrin.h>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#endif

namespace swiftware::hpp {

    void gemv(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp) {
        //TODO Implement GEMV: y = A * x + y
        int tileM = (Sp.TileSize1 > 0) ? Sp.TileSize1 : 32;
        int tileN = (Sp.TileSize2 > 0) ? Sp.TileSize2 : 32;

        #pragma omp parallel for
        for (int ii = 0; ii < m; ii += tileM) {
            int i_end = std::min(ii + tileM, m);
            for (int i = ii; i < i_end; ++i) {
                float sum = 0.0f;
                // Block over N dimension
                for (int jj = 0; jj < n; jj += tileN) {
                    int j_end = std::min(jj + tileN, n);
                    int p = jj;
                    __m256 sum_vec = _mm256_setzero_ps();
                    // SIMD over this tile
                    for (; p + 7 < j_end; p += 8) {
                        __m256 a_vec = _mm256_loadu_ps(&A[i * n + p]);
                        __m256 x_vec = _mm256_loadu_ps(&x[p]);
                        sum_vec = _mm256_fmadd_ps(a_vec, x_vec, sum_vec);
                    }
                    // Horizontal sum
                    float tmp[8];
                    _mm256_storeu_ps(tmp, sum_vec);
                    sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] +
                           tmp[4] + tmp[5] + tmp[6] + tmp[7];
                    // Handle leftovers
                    for (; p < j_end; ++p) {
                        sum += A[i * n + p] * x[p];
                    }
                }
                y[i] += sum;
            }
        }
    }

#ifdef USE_MKL
    void gemvMKL(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp) {
        // MKL GEMV: y = alpha*A*x + beta*y
        // Since y may have initial values, we use beta=1.0 to accumulate
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    m, n,
                    1.0f,        // alpha
                    A, n,        // A is m x n, leading dimension n
                    x, 1,        // x vector with increment 1
                    1.0f,        // beta (accumulate into y)
                    y, 1);       // y vector with increment 1
    }
#endif

}