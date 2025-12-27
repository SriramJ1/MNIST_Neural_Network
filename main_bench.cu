// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.



#include <nvbench/nvbench.cuh>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "gpu_utils.h"
#include "gpu_sparse_nn.cuh"


//#define CUDA_CHECK(x) swiftware::hpp::cuda_check((x), __FILE__, __LINE__)


void report_summary(nvbench::state& state)
{
    state.get_summary("nv/cold/time/gpu/min").remove_value("hide");
    state.get_summary("nv/cold/time/gpu/max").remove_value("hide");
    state.get_summary("nv/cold/time/gpu/mean").remove_value("hide");
    //state.get_summary("nv/cold/time/gpu/mean").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/mean").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/min").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/max").set_string("hide", "");
    state.get_summary("nv/cold/time/cpu/stdev/relative").set_string("hide", "");
    state.get_summary("nv/cold/sm_clock_rate/mean").remove_value("hide");
    state.get_summary("nv/cold/sm_clock_rate/scaling/percent").remove_value("hide");

}


void nvbench_gemm(nvbench::state& state)
{
    const size_t n = static_cast<size_t>(state.get_int64("n"));

    // Allocate host data
    std::vector<float> A(n*n, 1.0f);
    std::vector<float> B(n*n, 1.0f);
    std::vector<float> C(n*n, 0.0f);

    // Allocate device data
    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, n*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, n*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_d, n*n*sizeof(float)));


    CUDA_CHECK(cudaMemcpy(A_d, A.data(), n*n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), n*n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C.data(), n*n*sizeof(float), cudaMemcpyHostToDevice));

    //const int block = 256;
    //const int grid = static_cast<int>((n + block - 1) / block);
    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        // start timer
        timer.start();
        // TODO: launch your  kernel here
        swiftware::hpp::MM<<<gridDim, blockDim, 0, launch.get_stream()>>>(A_d, B_d, C_d, n, n, n);
        CUDA_CHECK(cudaGetLastError());
        // stop timer
        timer.stop();
    });

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));


    //TODO  compare to Ref with small epsilon

    report_summary(state);
}


void nvbench_gemv(nvbench::state& state)
{
    const size_t m = static_cast<size_t>(state.get_int64("m")); // rows
    const size_t n = static_cast<size_t>(state.get_int64("n")); // cols

    // Allocate host data
    std::vector<float> A(m * n, 1.0f);   // matrix m×n
    std::vector<float> x(n, 1.0f);       // vector length n
    std::vector<float> y(m, 0.0f);       // result length m

    // Allocate device data
    float *A_d, *x_d, *y_d;
    CUDA_CHECK(cudaMalloc(&A_d, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, m * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(A_d, A.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d, y.data(), m * sizeof(float), cudaMemcpyHostToDevice));

    // Configure kernel launch
    const int blockDimX = 256;
    const int gridDimX  = static_cast<int>((m + blockDimX - 1) / blockDimX);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer){
        timer.start();
        // GEMV kernel: each thread computes one row of y
        swiftware::hpp::MV<<<gridDimX, blockDimX, 0, launch.get_stream()>>>(A_d, x_d, y_d, m, n);
        CUDA_CHECK(cudaGetLastError());
        timer.stop();
    });

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));

    // TODO: compare y against reference with small epsilon

    report_summary(state);
}



void nvbench_spmv(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int n = static_cast<int>(state.get_int64("n"));
    const int sparsity = static_cast<int>(state.get_int64("sparsity"));

    // Build sparse matrix with target sparsity
    // sparsity is given as percentage (50, 60, 70, 80, 90, 95)
    std::vector<int> Ap(m+1);
    std::vector<int> Ai;
    std::vector<float> Ax;

    int nnz = 0;
    float sparsity_ratio = sparsity / 100.0f;
    
    for(int i=0; i<m; i++){
        Ap[i] = nnz;
        for(int j=0; j<n; j++){
            // Keep non-zero values based on sparsity percentage
            if((static_cast<float>(rand()) / RAND_MAX) > sparsity_ratio){
                Ai.push_back(j);
                Ax.push_back(1.0f);
                nnz++;
            }
        }
    }
    Ap[m] = nnz;

    std::vector<float> x(n, 1.0f);
    std::vector<float> y(m, 0.0f);
    std::vector<float> b(m, 0.0f);

    int *Ap_d, *Ai_d;
    float *Ax_d, *x_d, *y_d, *b_d;

    CUDA_CHECK(cudaMalloc(&Ap_d, (m+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&Ai_d, nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&Ax_d, nnz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x_d, n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, m*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b_d, m*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(Ap_d, Ap.data(), (m+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Ai_d, Ai.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Ax_d, Ax.data(), nnz*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d, y.data(), m*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(b_d, b.data(), m*sizeof(float), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (m + block - 1) / block;

    state.exec([&](nvbench::launch& launch) {
        swiftware::hpp::spmvKernel<<<grid, block, 0, launch.get_stream()>>>(Ap_d, Ai_d, Ax_d, x_d, y_d, b_d, m);
        CUDA_CHECK(cudaGetLastError());
    });

    CUDA_CHECK(cudaFree(Ap_d));
    CUDA_CHECK(cudaFree(Ai_d));
    CUDA_CHECK(cudaFree(Ax_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(b_d));

    report_summary(state);
}


void nvbench_spmm(nvbench::state& state)
{
    const int m = static_cast<int>(state.get_int64("m"));
    const int k = static_cast<int>(state.get_int64("k"));
    const int n = static_cast<int>(state.get_int64("n"));
    const int sparsity = static_cast<int>(state.get_int64("sparsity"));

    std::vector<float> B(k*n, 1.0f);
    std::vector<float> C(m*n, 0.0f);

    // Build sparse A with target sparsity
    std::vector<int> Ap(m+1);
    std::vector<int> Ai;
    std::vector<float> Ax;

    int nnz = 0;
    float sparsity_ratio = sparsity / 100.0f;
    
    for(int i=0; i<m; i++){
        Ap[i] = nnz;
        for(int j=0; j<k; j++){
            // Keep non-zero values based on sparsity percentage
            if((static_cast<float>(rand()) / RAND_MAX) > sparsity_ratio){
                Ai.push_back(j);
                Ax.push_back(1.0f);
                nnz++;
            }
        }
    }
    Ap[m] = nnz;

    int *Ap_d, *Ai_d;
    float *Ax_d, *B_d, *C_d;

    CUDA_CHECK(cudaMalloc(&Ap_d, (m+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&Ai_d, nnz*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&Ax_d, nnz*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&B_d, k*n*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&C_d, m*n*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(Ap_d, Ap.data(), (m+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Ai_d, Ai.data(), nnz*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Ax_d, Ax.data(), nnz*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B.data(), k*n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C_d, C.data(), m*n*sizeof(float), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (m + block - 1) / block;

    state.exec([&](nvbench::launch& launch) {
        swiftware::hpp::spmmKernel<<<grid, block, 0, launch.get_stream()>>>(
            Ap_d, Ai_d, Ax_d,
            B_d, C_d,
            m, n, k
        );
        CUDA_CHECK(cudaGetLastError());
    });

    CUDA_CHECK(cudaFree(Ap_d));
    CUDA_CHECK(cudaFree(Ai_d));
    CUDA_CHECK(cudaFree(Ax_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    report_summary(state);
}


NVBENCH_BENCH(nvbench_gemm).set_name("gemm").add_int64_axis("n", {256, 512, 1024, 2048, 4096});
NVBENCH_BENCH(nvbench_gemv).set_name("gemv").add_int64_axis("n", {256, 512, 1024, 2048, 4096});

// Original SPMV with 4096 (commented out for faster benchmarking)

NVBENCH_BENCH(nvbench_spmv)
  .set_name("gpu_spmv")
  .add_int64_axis("m", {512, 1024, 2048, 4096})
  .add_int64_axis("n", {512, 1024, 2048, 4096})
  .add_int64_axis("sparsity", {50, 60, 70, 80, 90, 95});


// SPMV with max size 1024 for faster execution
// NVBENCH_BENCH(nvbench_spmv)
//   .set_name("gpu_spmv")
//   .add_int64_axis("m", {512, 1024})
//   .add_int64_axis("n", {512, 1024})
//   .add_int64_axis("sparsity", {50, 60, 70, 80, 90, 95});

// Original SPMM with 4096 (commented out for faster benchmarking)

NVBENCH_BENCH(nvbench_spmm)
  .set_name("gpu_spmm")
  .add_int64_axis("m", {256, 512, 1024, 2048, 4096})
  .add_int64_axis("k", {256, 512, 1024, 2048, 4096})
  .add_int64_axis("n", {256, 512, 1024, 2048, 4096})
  .add_int64_axis("sparsity", {50, 60, 70, 80, 90, 95});


// // SPMM with max size 1024 for faster execution
// NVBENCH_BENCH(nvbench_spmm)
//   .set_name("gpu_spmm")
//   .add_int64_axis("m", {256, 512, 1024})
//   .add_int64_axis("k", {256, 512, 1024})
//   .add_int64_axis("n", {256, 512, 1024})
//   .add_int64_axis("sparsity", {50, 60, 70, 80, 90, 95});


