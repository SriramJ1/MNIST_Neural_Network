// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.



namespace swiftware::hpp {

// Test 1: Simple 2x2 identity-like sparse matrix
TEST(SPMV_Test, IdentityMatrix) {
    int m = 2, n = 2;
    std::vector<int> Ap = {0, 1, 2};           // row pointers
    std::vector<int> Ai = {0, 1};              // column indices
    std::vector<float> Ax = {1.0f, 1.0f};      // values
    std::vector<float> b = {2.0f, 3.0f};       // input vector
    std::vector<float> c(m, 0.0f);             // output vector
    
    ScheduleParams sp(1, 1);
    spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), b.data(), c.data(), sp);
    
    EXPECT_FLOAT_EQ(c[0], 2.0f);  // [1, 0] * [2, 3] = 2
    EXPECT_FLOAT_EQ(c[1], 3.0f);  // [0, 1] * [2, 3] = 3
}

// Test 2: 3x3 sparse matrix with diagonal elements
TEST(SPMV_Test, DiagonalMatrix) {
    int m = 3, n = 3;
    std::vector<int> Ap = {0, 1, 2, 3};        // row pointers
    std::vector<int> Ai = {0, 1, 2};           // column indices
    std::vector<float> Ax = {2.0f, 3.0f, 4.0f}; // values
    std::vector<float> b = {1.0f, 2.0f, 3.0f}; // input vector
    std::vector<float> c(m, 0.0f);             // output vector
    
    ScheduleParams sp(1, 1);
    spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), b.data(), c.data(), sp);
    
    EXPECT_FLOAT_EQ(c[0], 2.0f);   // 2 * 1 = 2
    EXPECT_FLOAT_EQ(c[1], 6.0f);   // 3 * 2 = 6
    EXPECT_FLOAT_EQ(c[2], 12.0f);  // 4 * 3 = 12
}

// Test 3: 4x4 sparse matrix with multiple non-zeros per row
TEST(SPMV_Test, SparseMatrix) {
    int m = 4, n = 4;
    // Sparse matrix with pattern:
    // [1, 0, 2, 0]
    // [0, 3, 0, 0]
    // [0, 0, 4, 1]
    // [5, 0, 0, 2]
    std::vector<int> Ap = {0, 2, 3, 5, 7};     // row pointers
    std::vector<int> Ai = {0, 2, 1, 2, 3, 0, 3}; // column indices
    std::vector<float> Ax = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 5.0f, 2.0f}; // values
    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f}; // input vector
    std::vector<float> c(m, 0.0f);             // output vector
    
    ScheduleParams sp(1, 1);
    spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), b.data(), c.data(), sp);
    
    EXPECT_FLOAT_EQ(c[0], 3.0f);   // 1 + 2 = 3
    EXPECT_FLOAT_EQ(c[1], 3.0f);   // 3
    EXPECT_FLOAT_EQ(c[2], 5.0f);   // 4 + 1 = 5
    EXPECT_FLOAT_EQ(c[3], 7.0f);   // 5 + 2 = 7
}

// Test 4: Single row sparse matrix
TEST(SPMV_Test, SingleRowMatrix) {
    int m = 1, n = 5;
    std::vector<int> Ap = {0, 2};              // row pointers
    std::vector<int> Ai = {0, 4};              // column indices
    std::vector<float> Ax = {2.0f, 3.0f};      // values
    std::vector<float> b = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}; // input vector
    std::vector<float> c(m, 0.0f);             // output vector
    
    ScheduleParams sp(1, 1);
    spmvCSR(m, n, Ap.data(), Ai.data(), Ax.data(), b.data(), c.data(), sp);
    
    EXPECT_FLOAT_EQ(c[0], 5.0f);   // 2 + 3 = 5
}

}