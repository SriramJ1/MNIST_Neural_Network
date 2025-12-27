// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "sparse_nn.h"
#include "def.h"
#include "utils.h"
#include "spmm.h"
#include "spmv.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>

namespace swiftware::hpp {

    // Helper: Create a simple sparse matrix for testing
    CSR createTestSparseMatrix(int m, int n, float sparsity = 0.9f) {
        CSR csr;
        csr.m = m;
        csr.n = n;
        csr.Ap.resize(m + 1);
        
        int nnz = 0;
        for (int i = 0; i < m; ++i) {
            csr.Ap[i] = nnz;
            for (int j = 0; j < n; ++j) {
                if ((i * n + j) % 10 == 0) {  // 10% non-zero
                    csr.Ai.push_back(j);
                    csr.Ax.push_back(1.0f);
                    nnz++;
                }
            }
        }
        csr.Ap[m] = nnz;
        return csr;
    }

    TEST(SparseNNTest, SpmmBasicTest) {
        DenseMatrix *X  = new DenseMatrix(1, 2); 
        X->data  = {1.0f, 2.0f};
        
        DenseMatrix *W1 = new DenseMatrix(1, 2); 
        W1->data = {1.0f, 1.0f};
        
        DenseMatrix *B1 = new DenseMatrix(1, 1); 
        B1->data = {0.0f};
        
        DenseMatrix *W2 = new DenseMatrix(1, 1); 
        W2->data = {2.0f};
        
        DenseMatrix *B2 = new DenseMatrix(1, 1); 
        B2->data = {0.0f};

        // Convert to sparse
        CSR W1_csr = denseToCSR(*W1);
        CSR W2_csr = denseToCSR(*W2);

        ScheduleParams Sp(-1, -1);
        DenseMatrix *pred = sparseNNSpmm(X, &W1_csr, &W2_csr, B1, B2, Sp);

        EXPECT_EQ(pred->m, 1);
        EXPECT_EQ(pred->n, 1);
        EXPECT_GE(pred->data[0], 0);
        EXPECT_LT(pred->data[0], 2);  // W2->m = 1

        delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
    }

    TEST(SparseNNTest, SpmvBasicTest) {
        DenseMatrix *X  = new DenseMatrix(1, 2); 
        X->data  = {1.0f, 2.0f};
        
        DenseMatrix *W1 = new DenseMatrix(1, 2); 
        W1->data = {1.0f, 1.0f};
        
        DenseMatrix *B1 = new DenseMatrix(1, 1); 
        B1->data = {0.0f};
        
        DenseMatrix *W2 = new DenseMatrix(1, 1); 
        W2->data = {2.0f};
        
        DenseMatrix *B2 = new DenseMatrix(1, 1); 
        B2->data = {0.0f};

        // Convert to sparse
        CSR W1_csr = denseToCSR(*W1);
        CSR W2_csr = denseToCSR(*W2);

        ScheduleParams Sp(-1, -1);
        DenseMatrix *pred = sparseNNSpmv(X, &W1_csr, &W2_csr, B1, B2, Sp);

        EXPECT_EQ(pred->m, 1);
        EXPECT_EQ(pred->n, 1);
        EXPECT_GE(pred->data[0], 0);
        EXPECT_LT(pred->data[0], 2);  // W2->m = 1

        delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
    }

    TEST(SparseNNTest, SpmmBatchTest) {
        DenseMatrix *X  = new DenseMatrix(2, 2); 
        X->data  = {1.0f, 2.0f, 4.0f, 1.0f};
        
        DenseMatrix *W1 = new DenseMatrix(2, 2); 
        W1->data = {1.0f, 0.5f, 0.5f, 1.0f};
        
        DenseMatrix *B1 = new DenseMatrix(1, 2); 
        B1->data = {0.0f, 0.0f};
        
        DenseMatrix *W2 = new DenseMatrix(2, 2); 
        W2->data = {1.0f, -1.0f, -1.0f, 1.0f};
        
        DenseMatrix *B2 = new DenseMatrix(1, 2); 
        B2->data = {0.0f, 0.0f};

        CSR W1_csr = denseToCSR(*W1);
        CSR W2_csr = denseToCSR(*W2);

        ScheduleParams Sp(-1, -1);
        DenseMatrix *pred = sparseNNSpmm(X, &W1_csr, &W2_csr, B1, B2, Sp);

        EXPECT_EQ(pred->m, 2);
        EXPECT_EQ(pred->n, 1);
        for (int i = 0; i < 2; ++i) {
            EXPECT_GE(pred->data[i], 0);
            EXPECT_LT(pred->data[i], W2->m);
        }

        delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
    }

    TEST(SparseNNTest, SpmvBatchTest) {
        DenseMatrix *X  = new DenseMatrix(2, 2); 
        X->data  = {1.0f, 2.0f, 4.0f, 1.0f};
        
        DenseMatrix *W1 = new DenseMatrix(2, 2); 
        W1->data = {1.0f, 0.5f, 0.5f, 1.0f};
        
        DenseMatrix *B1 = new DenseMatrix(1, 2); 
        B1->data = {0.0f, 0.0f};
        
        DenseMatrix *W2 = new DenseMatrix(2, 2); 
        W2->data = {1.0f, -1.0f, -1.0f, 1.0f};
        
        DenseMatrix *B2 = new DenseMatrix(1, 2); 
        B2->data = {0.0f, 0.0f};

        CSR W1_csr = denseToCSR(*W1);
        CSR W2_csr = denseToCSR(*W2);

        ScheduleParams Sp(-1, -1);
        DenseMatrix *pred = sparseNNSpmv(X, &W1_csr, &W2_csr, B1, B2, Sp);

        EXPECT_EQ(pred->m, 2);
        EXPECT_EQ(pred->n, 1);
        for (int i = 0; i < 2; ++i) {
            EXPECT_GE(pred->data[i], 0);
            EXPECT_LT(pred->data[i], W2->m);
        }

        delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
    }

    // ============ SPMV Tests ============
    TEST(SpMVTest, SpMVBasicTest) {
        // Test: y = A*x where A is sparse, simple 2x2 case
        // A = [[1, 0], [0, 1]] (diagonal), x = [2, 3]
        // Expected: y = [2, 3]
        
        int m = 2, n = 2;
        std::vector<int> Ap = {0, 1, 2};     // Row pointers
        std::vector<int> Ai = {0, 1};        // Column indices (diagonal)
        std::vector<float> Ax = {1.0f, 1.0f}; // Values
        std::vector<float> x = {2.0f, 3.0f};
        std::vector<float> y(m, 0.0f);
        
        ScheduleParams Sp(-1, -1);
        spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), x.data(), y.data(), Sp);
        
        EXPECT_NEAR(y[0], 2.0f, 1e-5);
        EXPECT_NEAR(y[1], 3.0f, 1e-5);
    }

    TEST(SpMVTest, SpMVMultiRowTest) {
        // Test: y = A*x with multiple non-zeros per row
        // A = [[1, 1], [1, 1]], x = [2, 3]
        // Expected: y = [5, 5]
        
        int m = 2, n = 2;
        std::vector<int> Ap = {0, 2, 4};     // Row 0: indices 0-1, Row 1: indices 2-3
        std::vector<int> Ai = {0, 1, 0, 1}; // Both rows have columns 0 and 1
        std::vector<float> Ax = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> x = {2.0f, 3.0f};
        std::vector<float> y(m, 0.0f);
        
        ScheduleParams Sp(-1, -1);
        spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), x.data(), y.data(), Sp);
        
        EXPECT_NEAR(y[0], 5.0f, 1e-5);
        EXPECT_NEAR(y[1], 5.0f, 1e-5);
    }

    // ============ SPMM Tests ============
    TEST(SpMMTest, SpMMBasicTest) {
        // Test: C = A*B where A is sparse
        // A = [[1, 0], [0, 1]] (diagonal 2x2), B = [[2, 3], [4, 5]] (2x2)
        // Expected: C = [[2, 3], [4, 5]]
        
        int m = 2, n = 2, k = 2;
        std::vector<int> Ap = {0, 1, 2};     // Row pointers
        std::vector<int> Ai = {0, 1};        // Column indices
        std::vector<float> Ax = {1.0f, 1.0f}; // Values
        std::vector<float> B = {2.0f, 3.0f, 4.0f, 5.0f}; // 2x2 dense matrix
        std::vector<float> C(m * n, 0.0f);
        
        ScheduleParams Sp(-1, -1);
        spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B.data(), C.data(), Sp);
        
        EXPECT_NEAR(C[0], 2.0f, 1e-5);  // Row 0: [2, 3]
        EXPECT_NEAR(C[1], 3.0f, 1e-5);
        EXPECT_NEAR(C[2], 4.0f, 1e-5);  // Row 1: [4, 5]
        EXPECT_NEAR(C[3], 5.0f, 1e-5);
    }

    TEST(SpMMTest, SpMMMultiNonzeroTest) {
        // Test: C = A*B with multiple non-zeros per row
        // A = [[1, 1], [1, 1]] (2x2), B = [[2, 0], [0, 3]] (2x2)
        // Expected: C = [[2, 3], [2, 3]]
        
        int m = 2, n = 2, k = 2;
        std::vector<int> Ap = {0, 2, 4};     // Both rows have 2 non-zeros
        std::vector<int> Ai = {0, 1, 0, 1}; // Column indices
        std::vector<float> Ax = {1.0f, 1.0f, 1.0f, 1.0f};
        std::vector<float> B = {2.0f, 0.0f, 0.0f, 3.0f}; // Diagonal B
        std::vector<float> C(m * n, 0.0f);
        
        ScheduleParams Sp(-1, -1);
        spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B.data(), C.data(), Sp);
        
        EXPECT_NEAR(C[0], 2.0f, 1e-5);  // Row 0: [2, 3]
        EXPECT_NEAR(C[1], 3.0f, 1e-5);
        EXPECT_NEAR(C[2], 2.0f, 1e-5);  // Row 1: [2, 3]
        EXPECT_NEAR(C[3], 3.0f, 1e-5);
    }
}

