// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.



#include "gpu_utils.h"      // For cuda_check
#include <cuda_runtime.h>
#include <vector>
#include <numeric>
#include "kernels.cuh"
#include "def.h"
#include <cmath>
#include <algorithm>
#include <cassert>
namespace swiftware::hpp {



    __global__ void bias_add_kernel(float* H, const float* B, int M, int N){
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M && col < N) {
            H[row * N + col] += B[col];
        }
    }

    __global__ void tanh_kernel(float* data, int size){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            data[i] = tanhf(data[i]);
        }
    }

    __global__ void sigmoid_kernel(float* data, int size){
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
            data[i] = 1.0f / (1.0f + expf(-data[i]));
        }
    }


    __global__ void argmax_kernel(const float* H, int M, int N, int* pred){
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < M) {
            int maxIdx = 0;
            float maxVal = H[row * N + 0];
            for (int c = 1; c < N; ++c) {
                float v = H[row * N + c];
                if (v > maxVal) {
                    maxVal = v;
                    maxIdx = c;
                }
            }
            pred[row] = maxIdx;
        }
    }

    __global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows && col < cols) {
            out[col * rows + row] = in[row * cols + col];
        }
    }


    DenseMatrix *dense_nn_gemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp){
        int batchSize = InData->m;
        int inDim = InData->n;
        int h1Dim = W1->m;
        int h2Dim = W2->m;

        float* dIn;
        float* dW1;
        float* dW2;
        float* dB1;
        float* dB2;

        float* dW1_raw;
        float* dW2_raw;

        int* dPred;
        float* dH1;
        float* dH2;

        CUDA_CHECK(cudaMalloc(&dIn, sizeof(float)*batchSize*inDim));
        CUDA_CHECK(cudaMalloc(&dW1_raw, sizeof(float)*h1Dim*inDim));
        CUDA_CHECK(cudaMalloc(&dW1, sizeof(float)*inDim*h1Dim));
        CUDA_CHECK(cudaMalloc(&dW2_raw, sizeof(float) * h2Dim * h1Dim));
        CUDA_CHECK(cudaMalloc(&dW2, sizeof(float)*h1Dim*h2Dim));
        CUDA_CHECK(cudaMalloc(&dB1, sizeof(float)*h1Dim));
        CUDA_CHECK(cudaMalloc(&dB2, sizeof(float)*h2Dim));

        CUDA_CHECK(cudaMalloc(&dPred, sizeof(int)*batchSize));
        CUDA_CHECK(cudaMalloc(&dH1, sizeof(float)*h1Dim*batchSize));
        CUDA_CHECK(cudaMalloc(&dH2, sizeof(float)*h2Dim*batchSize));

        CUDA_CHECK(cudaMemcpy(dIn, InData->data.data(), static_cast<size_t>(batchSize) * inDim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW1_raw, W1->data.data(), static_cast<size_t>(h1Dim) * inDim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dW2_raw, W2->data.data(), static_cast<size_t>(h2Dim) * h1Dim * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB1, B1->data.data(), static_cast<size_t>(h1Dim) * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB2, B2->data.data(), static_cast<size_t>(h2Dim) * sizeof(float), cudaMemcpyHostToDevice));

        const int blockX = Sp.TileSize1;
        const int blockY = Sp.TileSize2;
        const int vecBlock = Sp.VecTileSize;

        dim3 blockGemm(blockX, blockY);

        // Transpose W1 (h1Dim × inDim → inDim × h1Dim)
        dim3 gridW1((inDim + blockX - 1) / blockX, (h1Dim + blockY - 1) / blockY);
        transpose_kernel<<<gridW1, blockGemm>>>(dW1_raw, dW1, h1Dim, inDim);
        CUDA_CHECK(cudaGetLastError());

        // Transpose W2 (h2Dim × h1Dim → h1Dim × h2Dim)
        dim3 gridW2((h1Dim + blockX - 1) / blockX, (h2Dim + blockY - 1) / blockY);
        transpose_kernel<<<gridW2, blockGemm>>>(dW2_raw, dW2, h2Dim, h1Dim);
        CUDA_CHECK(cudaGetLastError());

        // Zero init outputs since MM accumulates
        cudaMemset(dH1, 0, sizeof(float) * h1Dim * batchSize);
        cudaMemset(dH2, 0, sizeof(float) * h2Dim * batchSize);

        dim3 gridGemm1((h1Dim + blockX - 1) / blockX, (batchSize + blockY - 1) / blockY);
        dim3 gridGemm2((h2Dim + blockX - 1) / blockX, (batchSize + blockY - 1) / blockY);

        dim3 blockVec(vecBlock);
        dim3 gridTanh((batchSize * h1Dim + vecBlock - 1) / vecBlock);
        dim3 gridSigm((batchSize * h2Dim + vecBlock - 1) / vecBlock);
        dim3 gridArgm((batchSize + vecBlock - 1) / vecBlock);


        swiftware::hpp::MM<<<gridGemm1, blockGemm>>>(dIn, dW1, dH1, batchSize, h1Dim, inDim);
        CUDA_CHECK(cudaGetLastError());

        bias_add_kernel<<<gridGemm1, blockGemm>>>(dH1, dB1, batchSize, h1Dim);
        CUDA_CHECK(cudaGetLastError());

        tanh_kernel<<<gridTanh, blockVec>>>(dH1, batchSize * h1Dim);
        CUDA_CHECK(cudaGetLastError());

        swiftware::hpp::MM<<<gridGemm2, blockGemm>>>(dH1, dW2, dH2, batchSize, h2Dim, h1Dim);
        CUDA_CHECK(cudaGetLastError());

        bias_add_kernel<<<gridGemm2, blockGemm>>>(dH2, dB2, batchSize, h2Dim);
        CUDA_CHECK(cudaGetLastError());

        sigmoid_kernel<<<gridSigm, blockVec>>>(dH2, batchSize * h2Dim);
        CUDA_CHECK(cudaGetLastError());

        argmax_kernel<<<gridArgm, blockVec>>>(dH2, batchSize, h2Dim, dPred);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> hPred(batchSize);
        CUDA_CHECK(cudaMemcpy(hPred.data(), dPred, static_cast<size_t>(batchSize) * sizeof(int), cudaMemcpyDeviceToHost));

        DenseMatrix* pred = new DenseMatrix(batchSize, 1);
        for (int i = 0; i < batchSize; ++i) {
            pred->data[i] = static_cast<float>(hPred[i]);
        }

        cudaFree(dIn);
        cudaFree(dW1);
        cudaFree(dW2);
        cudaFree(dB1);
        cudaFree(dB2);
        cudaFree(dH1);
        cudaFree(dH2);
        cudaFree(dPred);
        cudaFree(dW1_raw);
        cudaFree(dW2_raw);


        return pred;
    }


    DenseMatrix* dense_nn_gemv(DenseMatrix* InData, DenseMatrix* W1, DenseMatrix* W2, DenseMatrix* B1, DenseMatrix* B2, ScheduleParams Sp)
    {
    int batchSize = InData->m; // number of samples
    int inDim     = InData->n; // input dimension
    int h1Dim     = W1->m; // hidden layer size
    int h2Dim     = W2->m; // output layer size


    assert(W1->n == inDim);
    assert(W2->n == h1Dim);
    assert(static_cast<int>(B1->data.size()) == h1Dim);
    assert(static_cast<int>(B2->data.size()) == h2Dim);

    // Device buffers
    float *dX,*dW1,*dW2,*dB1,*dB2,*dH1,*dH2;
    int   *dPred;

    CUDA_CHECK(cudaMalloc(&dX,  sizeof(float) * batchSize * inDim));
    CUDA_CHECK(cudaMalloc(&dW1, sizeof(float) * h1Dim * inDim));
    CUDA_CHECK(cudaMalloc(&dW2, sizeof(float) * h2Dim * h1Dim));
    CUDA_CHECK(cudaMalloc(&dB1, sizeof(float) * h1Dim));
    CUDA_CHECK(cudaMalloc(&dB2, sizeof(float) * h2Dim));
    CUDA_CHECK(cudaMalloc(&dH1, sizeof(float) * batchSize * h1Dim));
    CUDA_CHECK(cudaMalloc(&dH2, sizeof(float) * batchSize * h2Dim));
    CUDA_CHECK(cudaMalloc(&dPred, sizeof(int) * batchSize));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(dX, InData->data.data(), sizeof(float) * batchSize * inDim, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW1, W1->data.data(),     sizeof(float) * h1Dim * inDim,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW2, W2->data.data(),     sizeof(float) * h2Dim * h1Dim,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB1, B1->data.data(),     sizeof(float) * h1Dim,            cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB2, B2->data.data(),     sizeof(float) * h2Dim,            cudaMemcpyHostToDevice));

    const int vecBlock = Sp.VecTileSize;
    dim3 blockBias(Sp.TileSize1, Sp.TileSize2);

    for (int i = 0; i < batchSize; ++i) {
        // Zero H1[i] before GEMV
        assert((i+1)*inDim <= static_cast<int>(InData->data.size()));
        assert((i+1)*h1Dim <= static_cast<int>(H1->data.size()) || true);
        CUDA_CHECK(cudaMemset(dH1 + i*h1Dim, 0, sizeof(float) * h1Dim));

        // H1[i] = W1 * X[i]
        int gridMV1 = (h1Dim + vecBlock - 1) / vecBlock;
        MV<<<gridMV1, vecBlock>>>(dW1, dX + i*inDim, dH1 + i*h1Dim, h1Dim, inDim);
        CUDA_CHECK(cudaGetLastError());

        // Bias add + tanh for H1[i]
        dim3 gridBias1((h1Dim + Sp.TileSize1 - 1) / Sp.TileSize1, 1);
        bias_add_kernel<<<gridBias1, blockBias>>>(dH1 + i*h1Dim, dB1, 1, h1Dim);
        CUDA_CHECK(cudaGetLastError());

        int gridTanh = (h1Dim + vecBlock - 1) / vecBlock;
        tanh_kernel<<<gridTanh, vecBlock>>>(dH1 + i*h1Dim, h1Dim);
        CUDA_CHECK(cudaGetLastError());

        // Zero H2[i] before GEMV
        assert((i+1)*h1Dim <= batchSize*h1Dim);
        assert((i+1)*h2Dim <= batchSize*h2Dim);
        CUDA_CHECK(cudaMemset(dH2 + i*h2Dim, 0, sizeof(float) * h2Dim));

        // H2[i] = W2 * H1[i]
        int gridMV2 = (h2Dim + vecBlock - 1) / vecBlock;
        MV<<<gridMV2, vecBlock>>>(dW2, dH1 + i*h1Dim, dH2 + i*h2Dim, h2Dim, h1Dim);
        CUDA_CHECK(cudaGetLastError());

        // Bias add for H2[i]
        dim3 gridBias2((h2Dim + Sp.TileSize1 - 1) / Sp.TileSize1, 1);
        bias_add_kernel<<<gridBias2, blockBias>>>(dH2 + i*h2Dim, dB2, 1, h2Dim);
        CUDA_CHECK(cudaGetLastError());
    }

    // Sigmoid across all H2
    int totalH2 = batchSize * h2Dim;
    int gridSigm = (totalH2 + vecBlock - 1) / vecBlock;
    sigmoid_kernel<<<gridSigm, vecBlock>>>(dH2, totalH2);
    CUDA_CHECK(cudaGetLastError());

    // Argmax per row
    int gridArgm = (batchSize + vecBlock - 1) / vecBlock;
    argmax_kernel<<<gridArgm, vecBlock>>>(dH2, batchSize, h2Dim, dPred);
    CUDA_CHECK(cudaGetLastError());

    // Copy predictions back
    std::vector<int> hPred(batchSize);
    CUDA_CHECK(cudaMemcpy(hPred.data(), dPred, sizeof(int) * batchSize, cudaMemcpyDeviceToHost));

    DenseMatrix* pred = new DenseMatrix(batchSize, 1);
    for (int i = 0; i < batchSize; ++i) pred->data[i] = static_cast<float>(hPred[i]);

    // Free device memory
    cudaFree(dX);
    cudaFree(dW1);
    cudaFree(dW2);
    cudaFree(dB1);
    cudaFree(dB2);
    cudaFree(dH1);
    cudaFree(dH2);
    cudaFree(dPred);

    return pred;
    }
}