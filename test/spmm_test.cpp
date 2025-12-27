// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


namespace swiftware::hpp {

// Test 1: Simple 2x2 identity-like sparse matrix times 2x2 dense matrix
TEST(SPMM_Test, IdentityTimes) {
    int m = 2, n = 2, k = 2;
    std::vector<int> Ap = {0, 1, 2};           // row pointers
    std::vector<int> Ai = {0, 1};              // column indices
    std::vector<float> Ax = {1.0f, 1.0f};      // values (identity)
    std::vector<float> B = {2.0f, 3.0f, 4.0f, 5.0f}; // B matrix (col-major)
    std::vector<float> C(m * n, 0.0f);         // output matrix
    
    ScheduleParams sp(1, 1);
    spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B.data(), C.data(), sp);
    
    // C should equal B (identity times B = B)
    EXPECT_FLOAT_EQ(C[0], 2.0f);
    EXPECT_FLOAT_EQ(C[1], 4.0f);
    EXPECT_FLOAT_EQ(C[2], 3.0f);
    EXPECT_FLOAT_EQ(C[3], 5.0f);
}

// Test 2: 3x3 diagonal sparse matrix times 3x3 dense matrix
TEST(SPMM_Test, DiagonalTimes) {
    int m = 3, n = 3, k = 3;
    std::vector<int> Ap = {0, 1, 2, 3};        // row pointers
    std::vector<int> Ai = {0, 1, 2};           // column indices (diagonal)
    std::vector<float> Ax = {2.0f, 3.0f, 4.0f}; // values
    std::vector<float> B = {1.0f, 2.0f, 3.0f,
                           4.0f, 5.0f, 6.0f,
                           7.0f, 8.0f, 9.0f};  // B matrix
    std::vector<float> C(m * n, 0.0f);         // output matrix
    
    ScheduleParams sp(1, 1);
    spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B.data(), C.data(), sp);
    
    // Diagonal scale: row i gets multiplied by Ax[i]
    EXPECT_FLOAT_EQ(C[0], 2.0f);   // 2 * 1
    EXPECT_FLOAT_EQ(C[1], 4.0f);   // 2 * 2
    EXPECT_FLOAT_EQ(C[2], 6.0f);   // 2 * 3
    EXPECT_FLOAT_EQ(C[3], 12.0f);  // 3 * 4
    EXPECT_FLOAT_EQ(C[4], 15.0f);  // 3 * 5
    EXPECT_FLOAT_EQ(C[5], 18.0f);  // 3 * 6
}

// Test 3: 2x3 sparse matrix times 3x2 dense matrix
TEST(SPMM_Test, RectangularMatrix) {
    int m = 2, n = 2, k = 3;
    // A is 2x3: [[1, 0, 2], [0, 3, 0]]
    std::vector<int> Ap = {0, 2, 3};           // row pointers
    std::vector<int> Ai = {0, 2, 1};           // column indices
    std::vector<float> Ax = {1.0f, 2.0f, 3.0f}; // values
    // B is 3x2: [[1, 2], [3, 4], [5, 6]]
    std::vector<float> B = {1.0f, 3.0f, 5.0f,
                           2.0f, 4.0f, 6.0f};  // B matrix
    std::vector<float> C(m * n, 0.0f);         // output matrix
    
    ScheduleParams sp(1, 1);
    spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B.data(), C.data(), sp);
    
    // C = A * B
    // C[0,0] = 1*1 + 2*5 = 11
    // C[0,1] = 1*2 + 2*6 = 14
    // C[1,0] = 3*3 = 9
    // C[1,1] = 3*4 = 12
    EXPECT_FLOAT_EQ(C[0], 11.0f);
    EXPECT_FLOAT_EQ(C[1], 9.0f);
    EXPECT_FLOAT_EQ(C[2], 14.0f);
    EXPECT_FLOAT_EQ(C[3], 12.0f);
}

// Test 4: 4x4 sparse matrix with multiple non-zeros per row
TEST(SPMM_Test, SparseMatrixMult) {
    int m = 4, n = 1, k = 4;
    // A is 4x4 sparse matrix
    // B is 4x1 dense vector (column vector)
    std::vector<int> Ap = {0, 2, 3, 5, 7};     // row pointers
    std::vector<int> Ai = {0, 2, 1, 2, 3, 0, 3}; // column indices
    std::vector<float> Ax = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 5.0f, 2.0f}; // values
    std::vector<float> B = {1.0f, 1.0f, 1.0f, 1.0f}; // B is 4x1 vector
    std::vector<float> C(m * n, 0.0f);         // output vector
    
    ScheduleParams sp(1, 1);
    spmmCSR(m, n, k, Ap.data(), Ai.data(), Ax.data(), B.data(), C.data(), sp);
    
    // Same as SPMV test
    EXPECT_FLOAT_EQ(C[0], 3.0f);
    EXPECT_FLOAT_EQ(C[1], 3.0f);
    EXPECT_FLOAT_EQ(C[2], 5.0f);
    EXPECT_FLOAT_EQ(C[3], 7.0f);
}

}