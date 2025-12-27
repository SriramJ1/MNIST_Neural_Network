// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "sparse_nn.h"
#include "spmm.h"
#include "spmv.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cassert>

namespace swiftware::hpp {

// Sigmoid helper (same as dense_nn.cpp)

static inline void sigmoid_vector(float* v, int n)
{
    for (int i = 0; i < n; ++i) 
        v[i] = 1.0f / (1.0f + std::exp(-v[i]));
}

// Argmax helper 
static inline int argmax_vec(const float* v, int n)
{
    int best = 0;
    float bestVal = v[0];

    for (int i = 1; i < n; ++i)
        if (v[i] > bestVal)
        {
            bestVal = v[i];
            best = i;
        }
    return best;
}

DenseMatrix* sparseNNSpmm(DenseMatrix* InData,CSR* W1, CSR* W2,DenseMatrix* B1, DenseMatrix* B2,ScheduleParams Sp)
{
    int batchSize  = InData->m;
    int inputDim   = InData->n;
    int hiddenDim  = W1->m;
    int outputDim  = W2->m;

    auto* pred = new DenseMatrix(batchSize, 1);

    std::vector<float> Z1(hiddenDim * batchSize, 0.0f);

    // SpMM layout: rows=m(A), cols=n(B) → we interpret X as column-major:
    // X[col][batch] indexing 
    spmmCSR(hiddenDim, batchSize, inputDim,W1->Ap.data(), W1->Ai.data(), W1->Ax.data(),InData->data.data(),Z1.data(),Sp); //B is X 

    // Bias + ReLU → H1
    std::vector<float> H1(batchSize * hiddenDim);

    for (int b = 0; b < batchSize; ++b)
        for (int h = 0; h < hiddenDim; ++h)
        {
            float v = Z1[h * batchSize + b] + B1->data[h]; // careful not to use indata
            H1[b * hiddenDim + h] = std::max(0.0f, v);
        }

    // Layer 2: H2 = sigmoid( W2 * H1 + B2 )
    // Using SpMM: W2[h2 x h1] * H1[h1 x B] --> Z2[h2 x B]

    std::vector<float> Z2(outputDim * batchSize, 0.0f);

    spmmCSR(outputDim, batchSize, hiddenDim,W2->Ap.data(), W2->Ai.data(), W2->Ax.data(),H1.data(),Z2.data(),Sp);

    // Bias + Sigmoid
    std::vector<float> H2(batchSize * outputDim);

    for (int b = 0; b < batchSize; ++b)
    {
        float* row = &H2[b * outputDim];
        for (int o = 0; o < outputDim; ++o)
            row[o] = Z2[o * batchSize + b] + B2->data[o];

        sigmoid_vector(row, outputDim);
    }

    // Argmax 

    for (int b = 0; b < batchSize; ++b)
        pred->data[b] = (float)argmax_vec(&H2[b * outputDim], outputDim);

    return pred;
}

DenseMatrix* sparseNNSpmv(DenseMatrix* InData,CSR* W1, CSR* W2,DenseMatrix* B1, DenseMatrix* B2,ScheduleParams Sp)
{
    int batchSize  = InData->m;
    int inputDim   = InData->n;
    int hiddenDim  = W1->m;
    int outputDim  = W2->m;

    auto* pred = new DenseMatrix(batchSize, 1);

    std::vector<float> h(hiddenDim);
    std::vector<float> z(outputDim);

    for (int s = 0; s < batchSize; ++s)
    {
        const float* x = &InData->data[s * inputDim];
  
        // Layer 1: h = ReLU(W1*x + B1)

        for (int i = 0; i < hiddenDim; ++i)
            h[i] = B1->data[i];

        spmvCSR(hiddenDim, inputDim,W1->Ap.data(), W1->Ai.data(), W1->Ax.data(),x, h.data(),Sp);

        for (int i = 0; i < hiddenDim; ++i)
            h[i] = std::max(0.0f, h[i]);

        // Layer 2: z = sigmoid(W2*h + B2)

        for (int i = 0; i < outputDim; ++i)
            z[i] = B2->data[i];

        spmvCSR(outputDim, hiddenDim,W2->Ap.data(), W2->Ai.data(), W2->Ax.data(),h.data(), z.data(),Sp);

        sigmoid_vector(z.data(), outputDim);

        // Argmax
        pred->data[s] = (float)argmax_vec(z.data(), outputDim);
    }

    return pred;
}

} // namespace swiftware::hpp