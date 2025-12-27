// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "dense_nn.h"
#include <cmath>
#include <algorithm>
#include <chrono>

#include <cmath>
#include <algorithm>
#include <cassert>
namespace swiftware::hpp {
    void sigmoid_matrix(DenseMatrix *M) {
        for (size_t i = 0; i < M->data.size(); i++) {
            M->data[i] = 1.0 / (1.0 + std::exp(-M->data[i]));
        }
    }

    int argmax(DenseMatrix *M, int row) {
        int bestIndex = 0;
        float bestValue = M->data[row * M->n];
        for (int j = 1; j < M->n; j++) {
            float v = M->data[row * M->n + j];
            if (v > bestValue) {
                bestIndex = j;
                bestValue = v;
            }
        }
        return bestIndex;
    }


    DenseMatrix* transpose(const swiftware::hpp::DenseMatrix* A) {
        auto* AT = new DenseMatrix(A->n, A->m); // AT: n × m
        for (int r = 0; r < A->m; ++r) {
            for (int c = 0; c < A->n; ++c) {
                AT->data[c * AT->n + r] = A->data[r * A->n + c];
            }
        }
        return AT;
    }


    // TODO Implement Dense NN with GEMM
    DenseMatrix *dense_nn_gemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        int batchSize = InData->m; // B
        int inDim = InData->n; // n
        int h1Dim = W1->m; // h1 (W1 is h1 × n)
        int h2Dim = W2->m; // h2 (W2 is h2 × h1)

        // Transpose weights W1T[n × h1] and W2T[h1 × h2]
        auto* W1T = transpose(W1);
        auto* W2T = transpose(W2);
    	
        auto* H1 = new DenseMatrix(batchSize, h1Dim); // B × h1
        auto* H2 = new DenseMatrix(batchSize, h2Dim); // B × h2
        auto* pred = new DenseMatrix(batchSize, 1);
        std::fill(H1->data.begin(), H1->data.end(), 0.0f);
        // H1 = X[B × n] · W1T[n × h1]
        gemm(batchSize, h1Dim, inDim, InData->data.data(), W1T->data.data(), H1->data.data(), Sp);

        // Bias add + tanh on H1
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < h1Dim; j++) {
                H1->data[i * h1Dim + j] += B1->data[j];
            }
        }
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < h1Dim; j++) {
                H1->data[i * h1Dim + j] = std::tanh(H1->data[i * h1Dim + j]);
            }
        }
        std::fill(H2->data.begin(), H2->data.end(), 0.0f);
        // H2 = H1[B × h1] · W2T[h1 × h2]
        gemm(batchSize, h2Dim, h1Dim, H1->data.data(), W2T->data.data(), H2->data.data(), Sp);

        // Bias add + sigmoid on H2
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < h2Dim; j++) {
                H2->data[i * h2Dim + j] += B2->data[j];
            }
        }
        sigmoid_matrix(H2);

        // Argmax per row
        for (int i = 0; i < batchSize; i++) {
            pred->data[i] = static_cast<float>(argmax(H2, i));
        }

        delete W1T;
        delete W2T;
        delete H1;
        delete H2;
        return pred;
    }

    //TODO: Implement Dense NN with GEMV
    DenseMatrix *dense_nn_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp){
        int batchSize = InData->m; // B
        int inDim = InData->n; // n
        int h1Dim = W1->m; // h1 (W1 is h1 × n)
        int h2Dim = W2->m; // h2 (W2 is h2 × h1)

        assert(W1->n == inDim);
        assert(W2->n == h1Dim);
        assert(static_cast<int>(B1->data.size()) == h1Dim);
        assert(static_cast<int>(B2->data.size()) == h2Dim);
        auto* H1 = new DenseMatrix(batchSize, h1Dim); // B × h1
        auto* H2 = new DenseMatrix(batchSize, h2Dim); // B × h2
        auto* pred = new DenseMatrix(batchSize, 1);

        for (int i = 0; i < batchSize; ++i) {
            // H1[i] = x · W1^T
            std::fill(&H1->data[i * h1Dim], &H1->data[(i + 1) * h1Dim], 0.0f);
            assert((i + 1) * inDim <= static_cast<int>(InData->data.size()));
            assert((i + 1) * h1Dim <= static_cast<int>(H1->data.size()));
            gemv(h1Dim, inDim, W1->data.data(), &InData->data[i * inDim], &H1->data[i * h1Dim], Sp);

            for (int j = 0; j < h1Dim; ++j) {
                H1->data[i * h1Dim + j] += B1->data[j];
                H1->data[i * h1Dim + j] = std::tanh(H1->data[i * h1Dim + j]);
            }
        }

        for (int i = 0; i < batchSize; ++i) {
            std::fill(&H2->data[i * h2Dim], &H2->data[(i + 1) * h2Dim], 0.0f);
            // H2[i] = h · W2^T
            assert((i + 1) * h1Dim <= static_cast<int>(H1->data.size()));
            assert((i + 1) * h2Dim <= static_cast<int>(H2->data.size()));
            gemv(h2Dim, h1Dim, W2->data.data(), &H1->data[i * h1Dim], &H2->data[i * h2Dim], Sp);

            for (int j = 0; j < h2Dim; ++j) {
                H2->data[i * h2Dim + j] += B2->data[j];
            }
        }

        sigmoid_matrix(H2);

        for (int i = 0; i < batchSize; ++i) {
            pred->data[i] = static_cast<float>(argmax(H2, i));
        }
        delete H1;
        delete H2;
        return pred;

    }

#ifdef USE_MKL

    DenseMatrix *dense_nn_mkl_gemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp) {
        int batchSize = InData->m; // B
        int inDim = InData->n; // n
        int h1Dim = W1->m; // h1 (W1 is h1 × n)
        int h2Dim = W2->m; // h2 (W2 is h2 × h1)

        // Transpose weights W1T[n × h1] and W2T[h1 × h2]
        auto* W1T = transpose(W1);
        auto* W2T = transpose(W2);

        auto* H1 = new DenseMatrix(batchSize, h1Dim); // B × h1
        auto* H2 = new DenseMatrix(batchSize, h2Dim); // B × h2
        auto* pred = new DenseMatrix(batchSize, 1);
        std::fill(H1->data.begin(), H1->data.end(), 0.0f);
        // H1 = X[B × n] · W1T[n × h1]
        gemmMKL(batchSize, h1Dim, inDim, InData->data.data(), W1T->data.data(), H1->data.data(), Sp);

        // Bias add + tanh on H1
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < h1Dim; j++) {
                H1->data[i * h1Dim + j] += B1->data[j];
            }
        }
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < h1Dim; j++) {
                H1->data[i * h1Dim + j] = std::tanh(H1->data[i * h1Dim + j]);
            }
        }
        std::fill(H2->data.begin(), H2->data.end(), 0.0f);
        // H2 = H1[B × h1] · W2T[h1 × h2]
        gemmMKL(batchSize, h2Dim, h1Dim, H1->data.data(), W2T->data.data(), H2->data.data(), Sp);

        // Bias add + sigmoid on H2
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < h2Dim; j++) {
                H2->data[i * h2Dim + j] += B2->data[j];
            }
        }
        sigmoid_matrix(H2);

        // Argmax per row
        for (int i = 0; i < batchSize; i++) {
            pred->data[i] = static_cast<float>(argmax(H2, i));
        }

        delete W1T;
        delete W2T;
        delete H1;
        delete H2;
        return pred;
    }

    DenseMatrix *dense_nn_mkl_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp)
        {
        int batchSize = InData->m;
        int inDim = InData->n;
        int h1Dim = W1->m; // W1 is h1 × n
        int h2Dim = W2->m; // W2 is h2 × h1

        assert(W1->n == inDim);
        assert(W2->n == h1Dim);
        assert(static_cast<int>(B1->data.size()) == h1Dim);
        assert(static_cast<int>(B2->data.size()) == h2Dim);

        DenseMatrix* H1 = new DenseMatrix(batchSize, h1Dim); // B × h1
        DenseMatrix* H2 = new DenseMatrix(batchSize, h2Dim); // B × h2
        DenseMatrix* pred = new DenseMatrix(batchSize, 1);

        for (int i = 0; i < batchSize; i++) {
            // H1[i] = x · W1^T
            std::fill(&H1->data[i * h1Dim], &H1->data[(i + 1) * h1Dim], 0.0f);
            gemvMKL(h1Dim, inDim, W1->data.data(), &InData->data[i * inDim], &H1->data[i * h1Dim], Sp);

            // Bias + tanh
            for (int j = 0; j < h1Dim; j++) {
                H1->data[i * h1Dim + j] += B1->data[j];
                H1->data[i * h1Dim + j] = std::tanh(H1->data[i * h1Dim + j]);
            }
        }

        for (int i = 0; i < batchSize; i++) {
            // H2[i] = h · W2^T
            std::fill(&H2->data[i * h2Dim], &H2->data[(i + 1) * h2Dim], 0.0f);
            gemvMKL(h2Dim, h1Dim, W2->data.data(), &H1->data[i * h1Dim], &H2->data[i * h2Dim], Sp);

            // Bias
            for (int j = 0; j < h2Dim; j++) {
                H2->data[i * h2Dim + j] += B2->data[j];
            }
        }

        // Sigmoid or softmax on H2
        sigmoid_matrix(H2);

        // Argmax for predictions
        for (int i = 0; i < batchSize; i++) {
            pred->data[i] = static_cast<float>(argmax(H2, i));
        }


        delete H1;
        delete H2;
        return pred;

    }
#endif

}
