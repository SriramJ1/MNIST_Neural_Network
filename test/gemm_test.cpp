// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "gemm.h"
#include <gtest/gtest.h>
#include <cmath>

namespace swiftware::hpp {
//TODO: Add more test

// Test basic gemm
TEST(GemmTest, BasicTest) {
    int m = 4, n = 4, k = 4;
    float A[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float B[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float C[16] = {0};
    
    ScheduleParams sp(2, 2);
    gemm(m, n, k, A, B, C, sp);
    
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}


TEST(GemmTest, NegativeValues) {
    int m = 2, n = 3, k = 2;
    float A[4] = {1,-2,3,-4}; // 2x2
    float B[6] = {-1,2,0,3,4,-5}; // 2x3
    float C[6] = {0};

    ScheduleParams sp(1,1);
    gemm(m,n,k,A,B,C,sp);

    // Manual calculation
    // C[0] = 1*-1 + -2*4 = -1 -8 = -9
    // C[1] = 1*2 + -2*-5 = 2 + 10 = 12
    // C[2] = 1*0 + -2*? = ?
    EXPECT_NEAR(C[0], -9.0f, 1e-5);
    EXPECT_NEAR(C[1], 12.0f, 1e-5);
    // continue for C[2..5] manually if desired
}

TEST(GemmTest, RectangularMatrices) {
    int m = 2, n = 3, k = 4;
    float A[8] = {1,2,3,4,5,6,7,8}; // 2x4
    float B[12] = {1,0,0,0,0,1,0,0,0,0,1,0}; // 4x3
    float C[6] = {0};

    ScheduleParams sp(1,1);
    gemm(m,n,k,A,B,C,sp);

    // Manually compute for correctness
    // row0: 1*1+2*0+3*0+4*0=1, 1*0+2*1+3*0+4*0=2, 1*0+2*0+3*1+4*0=3
    EXPECT_NEAR(C[0], 1.0f, 1e-5);
    EXPECT_NEAR(C[1], 2.0f, 1e-5);
    EXPECT_NEAR(C[2], 3.0f, 1e-5);
    EXPECT_NEAR(C[3], 5.0f, 1e-5); // row1: 5,6,7?
    EXPECT_NEAR(C[4], 6.0f, 1e-5);
    EXPECT_NEAR(C[5], 7.0f, 1e-5);
}
#ifdef USE_MKL
TEST(GemmTest, BasicTestMKL) {
    int m = 4, n = 4, k = 4;
    float A[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float B[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float C[16] = {0};

    ScheduleParams sp(2, 2);
    gemmMKL(m, n, k, A, B, C, sp);

    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(C[i], A[i], 1e-5);
    }
}

TEST(GemmTest, NegativeValuesMKL) {
    int m = 2, n = 3, k = 2;
    float A[4] = {1,-2,3,-4}; // 2x2
    float B[6] = {-1,2,0,3,4,-5}; // 2x3
    float C[6] = {0};

    ScheduleParams sp(1,1);
    gemmMKL(m,n,k,A,B,C,sp);

    // Manual calculation
    // C[0] = 1*-1 + -2*4 = -1 -8 = -9
    // C[1] = 1*2 + -2*-5 = 2 + 10 = 12
    // C[2] = 1*0 + -2*? = ?
    EXPECT_NEAR(C[0], -9.0f, 1e-5);
    EXPECT_NEAR(C[1], 12.0f, 1e-5);
    // continue for C[2..5] manually if desired
}
TEST(GemmTest, RectangularMatricesMKL) {
    int m = 2, n = 3, k = 4;
    float A[8] = {1,2,3,4,5,6,7,8}; // 2x4
    float B[12] = {1,0,0,0,0,1,0,0,0,0,1,0}; // 4x3
    float C[6] = {0};

    ScheduleParams sp(1,1);
    gemmMKL(m,n,k,A,B,C,sp);

    // Manually compute for correctness
    // row0: 1*1+2*0+3*0+4*0=1, 1*0+2*1+3*0+4*0=2, 1*0+2*0+3*1+4*0=3
    EXPECT_NEAR(C[0], 1.0f, 1e-5);
    EXPECT_NEAR(C[1], 2.0f, 1e-5);
    EXPECT_NEAR(C[2], 3.0f, 1e-5);
    EXPECT_NEAR(C[3], 5.0f, 1e-5); // row1: 5,6,7?
    EXPECT_NEAR(C[4], 6.0f, 1e-5);
    EXPECT_NEAR(C[5], 7.0f, 1e-5);
}
#endif


}

