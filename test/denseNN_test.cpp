// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "dense_nn.h"
#include "utils.h"
#include <gtest/gtest.h>
#include <cmath>
#include <iostream>


namespace swiftware::hpp {
 //TODO: Add more test


 TEST(DenseNNTest, GemmBasicTest) {
  DenseMatrix *X  = new DenseMatrix(1, 2); X->data  = {1.0f, 2.0f};
  DenseMatrix *W1 = new DenseMatrix(1, 2); W1->data = {1.0f, 1.0f};
  DenseMatrix *B1 = new DenseMatrix(1, 1); B1->data = {0.0f};
  DenseMatrix *W2 = new DenseMatrix(1, 1); W2->data = {2.0f};
  DenseMatrix *B2 = new DenseMatrix(1, 1); B2->data = {0.0f};

  ScheduleParams Sp(-1, -1);
  DenseMatrix *pred = dense_nn_gemm(X, W1, W2, B1, B2, Sp);

  EXPECT_EQ(pred->data[0], 0);

  delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
 }

 TEST(DenseNNTest, GemmMultiSampleTest) {
  DenseMatrix *X  = new DenseMatrix(2, 2); X->data  = {1.0f, 2.0f, 4.0f, 1.0f};
  DenseMatrix *W1 = new DenseMatrix(2, 2); W1->data = {1.0f, 0.5f, 0.5f, 1.0f};
  DenseMatrix *B1 = new DenseMatrix(1, 2); B1->data = {0.0f, 0.0f};
  DenseMatrix *W2 = new DenseMatrix(2, 2); W2->data = {1.0f, -1.0f, -1.0f, 1.0f};
  DenseMatrix *B2 = new DenseMatrix(1, 2); B2->data = {0.0f, 0.0f};

  ScheduleParams Sp(-1, -1);
  DenseMatrix *pred = dense_nn_gemm(X, W1, W2, B1, B2, Sp);

  EXPECT_EQ(pred->m, 2);
  EXPECT_EQ(pred->n, 1);
  EXPECT_GE(pred->data[0], 0);
  EXPECT_LT(pred->data[0], W2->m);
  EXPECT_GE(pred->data[1], 0);
  EXPECT_LT(pred->data[1], W2->m);

  delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
 }
// --- GEMV Tests ---
TEST(DenseNNTest, GemvBasicTest) {
  DenseMatrix *X  = new DenseMatrix(1, 2); X->data  = {1.0f, 2.0f};
  DenseMatrix *W1 = new DenseMatrix(1, 2); W1->data = {1.0f, 1.0f};
  DenseMatrix *B1 = new DenseMatrix(1, 1); B1->data = {0.0f};
  DenseMatrix *W2 = new DenseMatrix(1, 1); W2->data = {2.0f};
  DenseMatrix *B2 = new DenseMatrix(1, 1); B2->data = {0.0f};

  ScheduleParams Sp(-1, -1);
  DenseMatrix *pred = dense_nn_gemv(X, W1, W2, B1, B2, Sp);

  EXPECT_EQ(pred->data[0], 0);

  delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
}

TEST(DenseNNTest, GemvMultiSampleTest) {
  DenseMatrix *X  = new DenseMatrix(2, 2); X->data  = {1.0f, 2.0f, 4.0f, 1.0f};
  DenseMatrix *W1 = new DenseMatrix(2, 2); W1->data = {1.0f, 0.5f, 0.5f, 1.0f};
  DenseMatrix *B1 = new DenseMatrix(1, 2); B1->data = {0.0f, 0.0f};
  DenseMatrix *W2 = new DenseMatrix(2, 2); W2->data = {1.0f, -1.0f, -1.0f, 1.0f};
  DenseMatrix *B2 = new DenseMatrix(1, 2); B2->data = {0.0f, 0.0f};

  ScheduleParams Sp(-1, -1);
  DenseMatrix *pred = dense_nn_gemv(X, W1, W2, B1, B2, Sp);

  EXPECT_EQ(pred->m, 2);
  EXPECT_EQ(pred->n, 1);
  EXPECT_GE(pred->data[0], 0);
  EXPECT_LT(pred->data[0], W2->m);
  EXPECT_GE(pred->data[1], 0);
  EXPECT_LT(pred->data[1], W2->m);

  delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
}

#ifdef USE_MKL
// --- MKL GEMM Tests ---
TEST(DenseNNTest, MklGemmBasicTest) {
  DenseMatrix *X  = new DenseMatrix(1, 2); X->data  = {1.0f, 2.0f};
  DenseMatrix *W1 = new DenseMatrix(1, 2); W1->data = {1.0f, 1.0f};
  DenseMatrix *B1 = new DenseMatrix(1, 1); B1->data = {0.0f};
  DenseMatrix *W2 = new DenseMatrix(1, 1); W2->data = {2.0f};
  DenseMatrix *B2 = new DenseMatrix(1, 1); B2->data = {0.0f};

  ScheduleParams Sp(-1, -1);
  DenseMatrix *pred = dense_nn_mkl_gemm(X, W1, W2, B1, B2, Sp);

  EXPECT_EQ(pred->data[0], 0);

  delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
}

TEST(DenseNNTest, MklGemmMultiSampleTest) {
  DenseMatrix *X  = new DenseMatrix(2, 2); X->data  = {1.0f, 2.0f, 4.0f, 1.0f};
  DenseMatrix *W1 = new DenseMatrix(2, 2); W1->data = {1.0f, 0.5f, 0.5f, 1.0f};
  DenseMatrix *B1 = new DenseMatrix(1, 2); B1->data = {0.0f, 0.0f};
  DenseMatrix *W2 = new DenseMatrix(2, 2); W2->data = {1.0f, -1.0f, -1.0f, 1.0f};
  DenseMatrix *B2 = new DenseMatrix(1, 2); B2->data = {0.0f, 0.0f};

  ScheduleParams Sp(-1, -1);
  DenseMatrix *pred = dense_nn_mkl_gemm(X, W1, W2, B1, B2, Sp);

  EXPECT_EQ(pred->m, 2);
  EXPECT_EQ(pred->n, 1);
  EXPECT_GE(pred->data[0], 0);
  EXPECT_LT(pred->data[0], W2->m);
  EXPECT_GE(pred->data[1], 0);
  EXPECT_LT(pred->data[1], W2->m);

  delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
}

// --- MKL GEMV Tests ---
TEST(DenseNNTest, MklGemvBasicTest) {
  DenseMatrix *X  = new DenseMatrix(1, 2); X->data  = {1.0f, 2.0f};
  DenseMatrix *W1 = new DenseMatrix(1, 2); W1->data = {1.0f, 1.0f};
  DenseMatrix *B1 = new DenseMatrix(1, 1); B1->data = {0.0f};
  DenseMatrix *W2 = new DenseMatrix(1, 1); W2->data = {2.0f};
  DenseMatrix *B2 = new DenseMatrix(1, 1); B2->data = {0.0f};

  ScheduleParams Sp(-1, -1);
  DenseMatrix *pred = dense_nn_mkl_gemv(X, W1, W2, B1, B2, Sp);

  EXPECT_EQ(pred->data[0], 0);

  delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
}

TEST(DenseNNTest, MklGemvMultiSampleTest) {
  DenseMatrix *X  = new DenseMatrix(2, 2); X->data  = {1.0f, 2.0f, 4.0f, 1.0f};
  DenseMatrix *W1 = new DenseMatrix(2, 2); W1->data = {1.0f, 0.5f, 0.5f, 1.0f};
  DenseMatrix *B1 = new DenseMatrix(1, 2); B1->data = {0.0f, 0.0f};
  DenseMatrix *W2 = new DenseMatrix(2, 2); W2->data = {1.0f, -1.0f, -1.0f, 1.0f};
  DenseMatrix *B2 = new DenseMatrix(1, 2); B2->data = {0.0f, 0.0f};

  ScheduleParams Sp(-1, -1);
  DenseMatrix *pred = dense_nn_mkl_gemv(X, W1, W2, B1, B2, Sp);

  EXPECT_EQ(pred->m, 2);
  EXPECT_EQ(pred->n, 1);
  EXPECT_GE(pred->data[0], 0);
  EXPECT_LT(pred->data[0], W2->m);
  EXPECT_GE(pred->data[1], 0);
  EXPECT_LT(pred->data[1], W2->m);

  delete X; delete W1; delete W2; delete B1; delete B2; delete pred;
  }
#endif
}
