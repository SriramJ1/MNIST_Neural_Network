// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "utils.h"
#include "dense_nn.h"
#include <iostream>

#include "spmm.h"
#include "spmv.h"
#include "sparse_nn.h"


static void BM_GEMM(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0;
    }

    for (auto _: state) {
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete A;
    delete B;
    delete C;

}

static void BM_GEMV(benchmark::State &state) {
    int m  = state.range(0); // number of rows in A (batch size)
    int n  = state.range(1); // number of columns in A (input dimension)
    int t1 = state.range(2); // tile1 (or unused here)
    int t2 = state.range(3); // tile2 (or unused here)

    // A is m × n, B is n × 1 (vector), C is m × 1 (result)
    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *B = new swiftware::hpp::DenseMatrix(n, 1);
    auto *C = new swiftware::hpp::DenseMatrix(m, 1);

    // Fill with ones
    for (int i = 0; i < m * n; ++i) {
        A->data[i] = 1.0f;
    }
    for (int i = 0; i < n; ++i) {
        B->data[i] = 1.0f;
    }

    for (auto _ : state) {
        // GEMV: multiply A (m×n) by vector B (n×1) → C (m×1)
        swiftware::hpp::gemv(m, n, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }

    delete A;
    delete B;
    delete C;
}

// TODO add more benchmarks for MV, SPMV and SpMV

#ifdef USE_MKL
static void BM_GEMM_MKL(benchmark::State &state) {
    int m  = state.range(0);
    int n  = state.range(1);
    int k  = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);

    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);

    for (int i = 0; i < m * k; ++i) A->data[i] = 1.0f;
    for (int i = 0; i < k * n; ++i) B->data[i] = 1.0f;

    for (auto _ : state) {
        swiftware::hpp::gemmMKL(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }

    delete A; delete B; delete C;
}

static void BM_GEMV_MKL(benchmark::State &state) {
    int m  = state.range(0);
    int n  = state.range(1);
    int t1 = state.range(2);
    int t2 = state.range(3);

    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);

    for (int i = 0; i < m * n; ++i) A->data[i] = 1.0f;
    for (int i = 0; i < n; ++i) x->data[i] = 1.0f;

    for (auto _ : state) {
        swiftware::hpp::gemvMKL(m, n, A->data.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }

    delete A; delete x; delete y;
}
#endif






//

static void BM_SPMV(benchmark::State &state) {
    int m  = state.range(0); 
    int n  = state.range(1);  
    int t1 = state.range(2);  
    int t2 = state.range(3);

    
    swiftware::hpp::DenseMatrix dense(m, n);
    for (int i = 0; i < m * n; ++i)
        dense.data[i] = (i % 7 == 0) ? 1.0f : 0.0f;

    swiftware::hpp::CSR csr = swiftware::hpp::denseToCSR(dense);

    std::vector<float> x(n, 1.0f);
    std::vector<float> y(m, 0.0f);

    for (auto _ : state) {
        swiftware::hpp::spmvCSR(
            m, n,
            csr.Ap.data(),
            csr.Ai.data(),
            csr.Ax.data(),
            x.data(),
            y.data(),
            swiftware::hpp::ScheduleParams(t1, t2)
        );
    }
}

static void BM_SPMM(benchmark::State &state) {
    int m  = state.range(0);  // rows of A
    int n  = state.range(1);  // cols of B
    int k  = state.range(2);  // shared dim
    int t1 = state.range(3);
    int t2 = state.range(4);

    // Dense → CSR
    swiftware::hpp::DenseMatrix denseA(m, k);
    for (int i = 0; i < m * k; ++i)
        denseA.data[i] = (i % 9 == 0) ? 1.0f : 0.0f;

    swiftware::hpp::CSR csrA = swiftware::hpp::denseToCSR(denseA);

    // Dense matrix B
    std::vector<float> B(k * n, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    for (auto _ : state) {
        swiftware::hpp::spmmCSR(
            m, n, k,
            csrA.Ap.data(),
            csrA.Ai.data(),
            csrA.Ax.data(),
            B.data(),
            C.data(),
            swiftware::hpp::ScheduleParams(t1, t2)
        );
    }
}


//












static void BM_GEMM1(benchmark::State &state) {
    int m = state.range(0);
    int n = state.range(1);
    int k = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);
    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);
    for (int i = 0; i < m * k; ++i) {
        A->data[i] = 1.0;
    }
    for (int i = 0; i < k * n; ++i) {
        B->data[i] = 1.0;
    }

    for (auto _: state) {
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }
    delete A;
    delete B;
    delete C;

}

static void BM_GEMV1(benchmark::State &state) {
    int m  = state.range(0); // number of rows in A (batch size)
    int n  = state.range(1); // number of columns in A (input dimension)
    int t1 = state.range(2); // tile1 (or unused here)
    int t2 = state.range(3); // tile2 (or unused here)

    // A is m × n, B is n × 1 (vector), C is m × 1 (result)
    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *B = new swiftware::hpp::DenseMatrix(n, 1);
    auto *C = new swiftware::hpp::DenseMatrix(m, 1);

    // Fill with ones
    for (int i = 0; i < m * n; ++i) {
        A->data[i] = 1.0f;
    }
    for (int i = 0; i < n; ++i) {
        B->data[i] = 1.0f;
    }

    for (auto _ : state) {
        // GEMV: multiply A (m×n) by vector B (n×1) → C (m×1)
        swiftware::hpp::gemv(m, n, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }

    delete A;
    delete B;
    delete C;
}

// TODO add more benchmarks for MV, SPMV and SpMV

#ifdef USE_MKL
static void BM_GEMM_MKL1(benchmark::State &state) {
    int m  = state.range(0);
    int n  = state.range(1);
    int k  = state.range(2);
    int t1 = state.range(3);
    int t2 = state.range(4);

    auto *A = new swiftware::hpp::DenseMatrix(m, k);
    auto *B = new swiftware::hpp::DenseMatrix(k, n);
    auto *C = new swiftware::hpp::DenseMatrix(m, n);

    for (int i = 0; i < m * k; ++i) A->data[i] = 1.0f;
    for (int i = 0; i < k * n; ++i) B->data[i] = 1.0f;

    for (auto _ : state) {
        swiftware::hpp::gemmMKL(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }

    delete A; delete B; delete C;
}

static void BM_GEMV_MKL1(benchmark::State &state) {
    int m  = state.range(0);
    int n  = state.range(1);
    int t1 = state.range(2);
    int t2 = state.range(3);

    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);

    for (int i = 0; i < m * n; ++i) A->data[i] = 1.0f;
    for (int i = 0; i < n; ++i) x->data[i] = 1.0f;

    for (auto _ : state) {
        swiftware::hpp::gemvMKL(m, n, A->data.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(t1, t2));
    }

    delete A; delete x; delete y;
}
#endif
// For baseline and simd where tile sizes are not used
/*
BENCHMARK(BM_GEMM)
    ->Args({4096, 4096, 4096, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(10);
*/


#ifdef USE_MKL
// Register MKL benchmarks only if MKL is enabled
BENCHMARK(BM_GEMM_MKL)
    ->Args({128, 128, 128, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_GEMV_MKL)
    ->Args({128, 128, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);
#endif

// Sweep tile sizes for non‑MKL GEMM
BENCHMARK(BM_GEMM)
    ->Args({128, 128, 128, 8, 8})
    ->Args({128, 128, 128, 16, 16})
    ->Args({128, 128, 128, 32, 32})
    ->Args({128, 128, 128, 64, 64}) // baseline (no tiling)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

// Sweep tile sizes for non‑MKL GEMV
BENCHMARK(BM_GEMV)
    ->Args({128, 128, 8, 8})
    ->Args({128, 128, 16, 16})
    ->Args({128, 128, 32, 32})
    ->Args({128, 128, 64, 64}) // baseline (no tiling)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);









#ifdef USE_MKL
// Register MKL benchmarks only if MKL is enabled
BENCHMARK(BM_GEMM_MKL1)
    ->Args({128, 128, 128, -1, -1})
    ->Args({256, 256, 256, -1, -1})
    ->Args({512, 512, 512, -1, -1})
    ->Args({1024, 1024, 1024, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

BENCHMARK(BM_GEMV_MKL1)
    ->Args({128, 128, -1, -1})
    ->Args({256, 256,-1, -1})
    ->Args({512, 512, -1, -1})
    ->Args({1024, 1024, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);
#endif

// Sweep tile sizes for non‑MKL GEMM
BENCHMARK(BM_GEMM1)
    ->Args({128, 128, 128, 64, 64})
    ->Args({256, 256, 256, 64, 64})
    ->Args({512, 512, 512, 64, 64})
    ->Args({1024, 1024, 1024, 64, 64}) // baseline (no tiling)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

// Sweep tile sizes for non‑MKL GEMV
BENCHMARK(BM_GEMV1)
    ->Args({128, 128, 64, 64})
    ->Args({256, 256, 64, 64})
    ->Args({512, 512, 64, 64})
    ->Args({1024, 1024, 64, 64}) // baseline (no tiling)
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);


BENCHMARK(BM_SPMV)
    ->Args({128, 128, -1, -1})
    ->Args({256, 256, -1, -1})
    ->Args({512, 512, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_SPMM)
    ->Args({128, 128, 128, -1, -1})
    ->Args({256, 256, 256, -1, -1})
    ->Args({512, 512, 512, -1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);



BENCHMARK_MAIN();

