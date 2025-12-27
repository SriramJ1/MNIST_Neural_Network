// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "benchmark/benchmark.h"
#include "utils.h"
#include "dense_nn.h"
#include "def.h"
#include <iostream>
#include "sparse_nn.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include "spmv.h"
#include "spmm.h"

using namespace swiftware::hpp; 

template <typename DenseNNFunc>
static void BM_DENSENN(benchmark::State &state, DenseNNFunc nnFunc) {
/*
    auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
    auto *labels   = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
    auto *features = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

    // Extract labels and features
    for (int i = 0; i < mnistData->m; i++) {
        labels->data[i] = mnistData->data[i * mnistData->n]; // first column = label
        for (int j = 1; j < mnistData->n; j++) {
            features->data[i * (mnistData->n - 1) + (j - 1)] =
                mnistData->data[i * mnistData->n + j] / 255.0f;
        }
    }
*/
    auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
    auto *labels   = new swiftware::hpp::DenseMatrix(10, 1);
    auto *features = new swiftware::hpp::DenseMatrix(10, mnistData->n - 1);

    for (int i = 0; i < 10; ++i) { //first 10 samples
        // First column is the label
        labels->data[i] = mnistData->data[i * mnistData->n + 0];
        // Remaining columns are pixel values
        for (int j = 1; j < mnistData->n; ++j) {
            features->data[i * (mnistData->n - 1) + (j - 1)] =
                mnistData->data[i * mnistData->n + j] / 255.0f;
        }
    }

    auto *weightsOutput = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *weightsHidden = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *biasesHidden  = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *biasesOutput  = swiftware::hpp::readCSV("./data/model/biases_output.csv");

    swiftware::hpp::ScheduleParams scheduleParams(state.range(0), state.range(1));

    for (auto _ : state) {
        // Measure NN execution
        auto *pred = nnFunc(features, weightsHidden, weightsOutput, biasesHidden, biasesOutput, scheduleParams);

        state.PauseTiming(); // exclude accuracy calc
        int correctPredictions = 0;
        for (int i = 0; i < labels->m; i++) {
            int predicted = static_cast<int>(pred->data[i]);
            int actual    = static_cast<int>(labels->data[i]);
            //std::cout << "Sample " << i  << " -> Predicted: " << predicted  << ", Label: " << actual << std::endl;
            if (predicted == actual) {
                correctPredictions++;
            }
        }
        double accuracy = static_cast<double>(correctPredictions) / labels->m;
        state.counters["Accuracy"] = accuracy;
        delete pred;
        state.ResumeTiming();
    }

    delete mnistData;
    delete labels;
    delete features;
    delete weightsOutput;
    delete weightsHidden;
    delete biasesHidden;
    delete biasesOutput;
}




template <typename DenseNNFunc>
static void BM_DENSENN1(benchmark::State &state, DenseNNFunc nnFunc) {

    auto *mnistData = swiftware::hpp::readCSV("./data/mnist_train.csv", true);
    auto *labels   = new swiftware::hpp::DenseMatrix(mnistData->m, 1);
    auto *features = new swiftware::hpp::DenseMatrix(mnistData->m, mnistData->n - 1);

    // Extract labels and features
    for (int i = 0; i < mnistData->m; i++) {
        labels->data[i] = mnistData->data[i * mnistData->n]; // first column = label
        for (int j = 1; j < mnistData->n; j++) {
            features->data[i * (mnistData->n - 1) + (j - 1)] =
                mnistData->data[i * mnistData->n + j] / 255.0f;
        }
    }



    auto *weightsOutput = swiftware::hpp::readCSV("./data/model/weights_output.csv");
    auto *weightsHidden = swiftware::hpp::readCSV("./data/model/weights_hidden.csv");
    auto *biasesHidden  = swiftware::hpp::readCSV("./data/model/biases_hidden.csv");
    auto *biasesOutput  = swiftware::hpp::readCSV("./data/model/biases_output.csv");

    swiftware::hpp::ScheduleParams scheduleParams(state.range(0), state.range(1));

    for (auto _ : state) {
        // Measure NN execution
        auto *pred = nnFunc(features, weightsHidden, weightsOutput, biasesHidden, biasesOutput, scheduleParams);

        state.PauseTiming(); // exclude accuracy calc
        int correctPredictions = 0;
        for (int i = 0; i < labels->m; i++) {
            int predicted = static_cast<int>(pred->data[i]);
            int actual    = static_cast<int>(labels->data[i]);
            //std::cout << "Sample " << i  << " -> Predicted: " << predicted  << ", Label: " << actual << std::endl;
            if (predicted == actual) {
                correctPredictions++;
            }
        }
        double accuracy = static_cast<double>(correctPredictions) / labels->m;
        state.counters["Accuracy"] = accuracy;
        delete pred;
        state.ResumeTiming();
    }

    delete mnistData;
    delete labels;
    delete features;
    delete weightsOutput;
    delete weightsHidden;
    delete biasesHidden;
    delete biasesOutput;
}




//
// ---------------------------------------------------
//  Sparse Matrix–Vector Benchmark (SPMV)
// ---------------------------------------------------
//
static void BM_SPMV(benchmark::State &state)
{
    int m = state.range(0);   // rows
    int n = state.range(1);   // cols

    // ----- Create a sparse-ish dense matrix -----
    DenseMatrix A_dense(m, n);
    for (int i = 0; i < m * n; ++i)
        A_dense.data[i] = (i % 7 == 0) ? 1.0f : 0.0f;

    // Convert to CSR
    CSR A = denseToCSR(A_dense);

    // Vector x and output y
    std::vector<float> x(n, 1.0f);
    std::vector<float> y(m, 0.0f);

    ScheduleParams Sp(-1, -1);   // no tiling used

    // ----- Benchmark Loop -----
    for (auto _ : state)
    {
        spmvCSR(
            m,
            n,
            A.Ap.data(),
            A.Ai.data(),
            A.Ax.data(),
            x.data(),
            y.data(),
            Sp
        );
    }
}

//
// ---------------------------------------------------
//  Sparse Matrix–Matrix Benchmark (SPMM)
// ---------------------------------------------------
//
static void BM_SPMM(benchmark::State &state)
{
    int m = state.range(0);   // rows of A
    int n = state.range(1);   // cols of B
    int k = state.range(2);   // shared dimension

    // ----- Create sparse-ish dense A -----
    DenseMatrix A_dense(m, k);
    for (int i = 0; i < m * k; ++i)
        A_dense.data[i] = (i % 9 == 0) ? 1.0f : 0.0f;

    // Convert to CSR
    CSR A = denseToCSR(A_dense);

    // Dense B and output C
    std::vector<float> B(k * n, 1.0f);
    std::vector<float> C(m * n, 0.0f);

    ScheduleParams Sp(-1, -1);

    // ----- Benchmark Loop -----
    for (auto _ : state)
    {
        spmmCSR(
            m,
            n,
            k,
            A.Ap.data(),
            A.Ai.data(),
            A.Ax.data(),
            B.data(),
            C.data(),
            Sp
        );
    }
}

//
// ---------------------------------------------------
//  Benchmark Registrations
// ---------------------------------------------------
//
BENCHMARK(BM_SPMV)
    ->Args({128, 128})
    ->Args({256, 256})
    ->Args({512, 512})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_SPMM)
    ->Args({128, 128, 128})
    ->Args({256, 256, 256})
    ->Args({512, 512, 512})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);



/*
// For baseline and simd where tile sizes are not used
BENCHMARK(BM_DENSENN)
    ->Args({32, 32})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);
*/

// Wrappers for each implementation
static void BM_DENSE_GEMM(benchmark::State &state) {
    BM_DENSENN(state, swiftware::hpp::dense_nn_gemm);
}

static void BM_DENSE_GEMV(benchmark::State &state) {
    BM_DENSENN(state, swiftware::hpp::dense_nn_gemv);
}

#ifdef USE_MKL
static void BM_DENSE_MKL_GEMM(benchmark::State &state) {
    BM_DENSENN(state, swiftware::hpp::dense_nn_mkl_gemm);
}

static void BM_DENSE_MKL_GEMV(benchmark::State &state) {
    BM_DENSENN(state, swiftware::hpp::dense_nn_mkl_gemv);
}
#endif





static void BM_DENSE_GEMM1(benchmark::State &state) {
    BM_DENSENN1(state, swiftware::hpp::dense_nn_gemm);
}

static void BM_DENSE_GEMV1(benchmark::State &state) {
    BM_DENSENN1(state, swiftware::hpp::dense_nn_gemv);
}

#ifdef USE_MKL
static void BM_DENSE_MKL_GEMM1(benchmark::State &state) {
    BM_DENSENN1(state, swiftware::hpp::dense_nn_mkl_gemm);
}

static void BM_DENSE_MKL_GEMV1(benchmark::State &state) {
    BM_DENSENN1(state, swiftware::hpp::dense_nn_mkl_gemv);
}
#endif

// Register benchmarks
BENCHMARK(BM_DENSE_GEMM)
    ->Args({64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_DENSE_GEMV)
    ->Args({64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

#ifdef USE_MKL
BENCHMARK(BM_DENSE_MKL_GEMM)
    ->Args({-1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

BENCHMARK(BM_DENSE_MKL_GEMV)
    ->Args({-1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(5);

#endif




// Register benchmarks
BENCHMARK(BM_DENSE_GEMM1)
    ->Args({64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

BENCHMARK(BM_DENSE_GEMV1)
    ->Args({64, 64})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

#ifdef USE_MKL
BENCHMARK(BM_DENSE_MKL_GEMM1)
    ->Args({-1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

BENCHMARK(BM_DENSE_MKL_GEMV1)
    ->Args({-1, -1})
    ->Unit(benchmark::kMicrosecond)
    ->Iterations(1)
    ->Repetitions(3);

#endif

BENCHMARK_MAIN();
