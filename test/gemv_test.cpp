// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.


#include "gemv.h"
#include <gtest/gtest.h>
#include <cmath>


namespace swiftware::hpp {
    //TODO: Add more test

// Test basic gemv
TEST(GemvTest, BasicTest) {
    int m = 4, n = 4;
    float A[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float x[4] = {1,2,3,4};
    float y[4] = {0};
    
    ScheduleParams sp(2, 2);
    gemv(m, n, A, x, y, sp);
    
    EXPECT_NEAR(y[0], 1.0f, 1e-5);
    EXPECT_NEAR(y[1], 2.0f, 1e-5);
    EXPECT_NEAR(y[2], 3.0f, 1e-5);
    EXPECT_NEAR(y[3], 4.0f, 1e-5);
}


TEST(GemvTest, NegativeValues) {
    int m = 2, n = 3;
    float A[6] = {-1, 2, -3, 4, -5, 6};
    float x[3] = {1, -1, 2};
    float y[2] = {0};

    ScheduleParams sp(1,1);
    gemv(m, n, A, x, y, sp);

    // Manual calculation:
    // row 0: (-1*1) + (2*-1) + (-3*2) = -1 -2 -6 = -9
    // row 1: (4*1) + (-5*-1) + (6*2) = 4 +5 +12 = 21
    EXPECT_NEAR(y[0], -9.0f, 1e-5);
    EXPECT_NEAR(y[1], 21.0f, 1e-5);
}

TEST(GemvTest, RectangularMatrix) {
    int m = 2, n = 4;
    float A[8] = {1,2,3,4,5,6,7,8};
    float x[4] = {1,1,1,1};
    float y[2] = {0};

    ScheduleParams sp(1,1);
    gemv(m, n, A, x, y, sp);

    // row 0: 1+2+3+4=10, row1: 5+6+7+8=26
    EXPECT_NEAR(y[0], 10.0f, 1e-5);
    EXPECT_NEAR(y[1], 26.0f, 1e-5);
}


TEST(GemvTest, TallMatrix) {
    int m = 4, n = 2;
    float A[8] = {1,2,3,4,5,6,7,8};
    float x[2] = {1, -1};
    float y[4] = {0};

    ScheduleParams sp(1,1);
    gemv(m, n, A, x, y, sp);

    // row 0: 1*1 + 2*-1 = -1
    // row 1: 3*1 + 4*-1 = -1
    // row 2: 5*1 + 6*-1 = -1
    // row 3: 7*1 + 8*-1 = -1
    for(int i=0;i<4;i++)
        EXPECT_NEAR(y[i], -1.0f, 1e-5);
}
#ifdef USE_MKL
TEST(GemvTest, BasicTestMKL) {
    int m = 4, n = 4;
    float A[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    float x[4] = {1,2,3,4};
    float y[4] = {0};

    ScheduleParams sp(2, 2);
    gemvMKL(m, n, A, x, y, sp);

    EXPECT_NEAR(y[0], 1.0f, 1e-5);
    EXPECT_NEAR(y[1], 2.0f, 1e-5);
    EXPECT_NEAR(y[2], 3.0f, 1e-5);
    EXPECT_NEAR(y[3], 4.0f, 1e-5);
}

TEST(GemvTest, NegativeValuesMKL) {
    int m = 2, n = 3;
    float A[6] = {-1, 2, -3, 4, -5, 6};
    float x[3] = {1, -1, 2};
    float y[2] = {0};

    ScheduleParams sp(1,1);
    gemvMKL(m, n, A, x, y, sp);

    // Manual calculation:
    // row 0: (-1*1) + (2*-1) + (-3*2) = -1 -2 -6 = -9
    // row 1: (4*1) + (-5*-1) + (6*2) = 4 +5 +12 = 21
    EXPECT_NEAR(y[0], -9.0f, 1e-5);
    EXPECT_NEAR(y[1], 21.0f, 1e-5);
}

TEST(GemvTest, RectangularMatrixMKL) {
    int m = 2, n = 4;
    float A[8] = {1,2,3,4,5,6,7,8};
    float x[4] = {1,1,1,1};
    float y[2] = {0};

    ScheduleParams sp(1,1);
    gemvMKL(m, n, A, x, y, sp);

    // row 0: 1+2+3+4=10, row1: 5+6+7+8=26
    EXPECT_NEAR(y[0], 10.0f, 1e-5);
    EXPECT_NEAR(y[1], 26.0f, 1e-5);
}


TEST(GemvTest, TallMatrixMKL) {
    int m = 4, n = 2;
    float A[8] = {1,2,3,4,5,6,7,8};
    float x[2] = {1, -1};
    float y[4] = {0};

    ScheduleParams sp(1,1);
    gemvMKL(m, n, A, x, y, sp);

    // row 0: 1*1 + 2*-1 = -1
    // row 1: 3*1 + 4*-1 = -1
    // row 2: 5*1 + 6*-1 = -1
    // row 3: 7*1 + 8*-1 = -1
    for(int i=0;i<4;i++)
        EXPECT_NEAR(y[i], -1.0f, 1e-5);
}
#endif

}

