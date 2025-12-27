// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gpu_sparse_nn.cuh"
#include "gpu_utils.h"
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "gpu_dense_nn.cuh" //borrowing functions


namespace swiftware::hpp {

// Global namespacce kyu use kia reason karo
//  ReLU kernel (GPU)

__global__ void relu_kernel(float* data, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        data[i] = (data[i] > 0.0f ? data[i] : 0.0f); //fasterrr than if
    }
}



// Argmax kernel — 1 thread per NN output row (batch=1)

__global__ void argmax_kernel_sparse(const float* z, int n, int* pred)
{
    float bestVal = z[0];
    int bestIdx = 0;

    for (int i = 1; i < n; i++)
    {
        float v = z[i];
        if (v > bestVal)
        {
            bestVal = v;
            bestIdx = i;
        }
    }
    *pred = bestIdx;
}


// SPMV kernel implementation: one thread per row
// out[row] = bias[row] + sum(A[row,col] * x[col])
__global__ void spmvKernel(
    const int *Ap, const int *Ai, const float *Ax,
    const float *x, float *out, const float *bias, int m)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    float sum = bias[row];
    int start = Ap[row];
    int end   = Ap[row + 1];

    for (int idx = start; idx < end; idx++)
    {
        int col = Ai[idx];
        sum += Ax[idx] * x[col];
    }
    out[row] = sum;
}


// SPMM kernel implementation: Sparse A @ Dense B = Dense C
// A is m x k (sparse CSR), B is k x n (dense), C is m x n (dense)
__global__ void spmmKernel(
    const int *Ap, const int *Ai, const float *Ax,
    const float *B, float *C, int m, int n, int k)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    // Initialize C[row, :] to 0
    for (int j = 0; j < n; ++j)
        C[row * n + j] = 0.0f;

    // Iterate through non-zeros in A[row, :]
    int start = Ap[row];
    int end   = Ap[row + 1];

    for (int idx = start; idx < end; ++idx)
    {
        int colA = Ai[idx];
        float valA = Ax[idx];

        // Add valA * B[colA, :] to C[row, :]
        const float *Brow = B + colA * n;
        float *Crow = C + row * n;

        for (int j = 0; j < n; ++j)
        {
            Crow[j] += valA * Brow[j];
        }
    }
}


// Full Sparse NN forward pass on GPU (batch = 1)

DenseMatrix* sparseNNGPU(DenseMatrix *InData,CSR *W1, CSR *W2,DenseMatrix *B1, DenseMatrix *B2)
{
    int inputDim  = InData->n;
    int hiddenDim = W1->m;
    int outputDim = W2->m;

    DenseMatrix *pred = new DenseMatrix(1, 1);

   
    float *d_x, *d_h, *d_z, *d_b1, *d_b2;
    int   *d_Ap1, *d_Ai1, *d_Ap2, *d_Ai2;
    float *d_Ax1, *d_Ax2;
    int *dPred;

    int nnz1 = W1->Ax.size();
    int nnz2 = W2->Ax.size();

    cudaMalloc(&d_x, inputDim  * sizeof(float));
    cudaMalloc(&d_h, hiddenDim * sizeof(float));
    cudaMalloc(&d_z, outputDim * sizeof(float));

    cudaMalloc(&d_b1,hiddenDim * sizeof(float));
    cudaMalloc(&d_b2,outputDim * sizeof(float));

    cudaMalloc(&dPred, sizeof(int));

    cudaMalloc(&d_Ap1, (hiddenDim + 1) * sizeof(int));
    cudaMalloc(&d_Ai1, nnz1 * sizeof(int));
    cudaMalloc(&d_Ax1, nnz1 * sizeof(float));

    cudaMalloc(&d_Ap2, (outputDim + 1) * sizeof(int));
    cudaMalloc(&d_Ai2, nnz2 * sizeof(int));
    cudaMalloc(&d_Ax2, nnz2 * sizeof(float));

    // do not use . size use . data 
    cudaMemcpy(d_x,  InData->data.data(), inputDim  * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, B1->data.data(), hiddenDim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, B2->data.data(),  outputDim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_Ap1, W1->Ap.data(), (hiddenDim + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ai1, W1->Ai.data(), nnz1 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ax1, W1->Ax.data(), nnz1 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_Ap2, W2->Ap.data(), (outputDim + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ai2, W2->Ai.data(), nnz2 * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_Ax2, W2->Ax.data(), nnz2 * sizeof(float), cudaMemcpyHostToDevice);

    // Layer 1: h = ReLU( W1 * x + b1 )
    
    int block = 128;
    int grid1 = (hiddenDim + block - 1) / block;

    spmvKernel<<<grid1, block>>>(d_Ap1, d_Ai1, d_Ax1,d_x,d_h,d_b1,hiddenDim);

    // ReLU on GPU (malli check cheye)
    relu_kernel<<<grid1, block>>>(d_h, hiddenDim);

    // Layer 2: z = sigmoid( W2 * h + b2 )

    int grid2 = (outputDim + block - 1) / block;

    spmvKernel<<<grid2, block>>>(d_Ap2, d_Ai2, d_Ax2,d_h,d_z,d_b2,outputDim);

    // Sigmoid
    sigmoid_kernel<<<grid2, block>>>(d_z, outputDim);


    // Argmax — GPU version

    argmax_kernel_sparse<<<1,1>>>(d_z, outputDim, dPred);

    int hPred;
    cudaMemcpy(&hPred, dPred, sizeof(int), cudaMemcpyDeviceToHost);
    pred->data[0] = (float)hPred;

    // Cleanup

    cudaFree(d_x); cudaFree(d_h); cudaFree(d_z);
    cudaFree(d_b1); cudaFree(d_b2);
    cudaFree(d_Ap1); cudaFree(d_Ai1); cudaFree(d_Ax1);
    cudaFree(d_Ap2); cudaFree(d_Ai2); cudaFree(d_Ax2);
    cudaFree(dPred);

    return pred;
}

} // namespace swiftware::hpp
