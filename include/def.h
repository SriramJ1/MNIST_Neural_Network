// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.



#ifndef PROJECT_DENSE_MATMUL_DEF_H
#define PROJECT_DENSE_MATMUL_DEF_H

#include <vector>


namespace swiftware::hpp {

 struct ScheduleParams {
  int TileSize1;
  int TileSize2;
  int VecTileSize;
  // Allow callers to pass only two tile sizes; VecTileSize defaults to 1
  ScheduleParams(int TileSize1, int TileSize2, int VecTileSize = 1) : TileSize1(TileSize1), TileSize2(TileSize2), VecTileSize(VecTileSize) {}
 };

 // please do not change the following struct
 struct DenseMatrix{
  int m;
  int n;
  std::vector<float> data;
  DenseMatrix(int m, int n): m(m), n(n), data(m*n){}
 };

 //I added below 
 struct CSR {
    int m;                     // number of rows
    int n;                     // number of cols
    std::vector<int> Ap;       // row_ptr (size m+1)
    std::vector<int> Ai;       // col_idx (size nnz)
    std::vector<float> Ax;     // values (size nnz)
};
CSR denseToCSR(const DenseMatrix &dense);
//{
    // int m = dense.m;
    // int n = dense.n;

    // CSR csr;
    // csr.m = m;
    // csr.n = n;
    // csr.Ap.resize(m + 1);

    // for (int i = 0; i < m; i++)
    // {
    //     csr.Ap[i] = csr.Ax.size();   // row start index

    //     for (int j = 0; j < n; j++)
    //     {
    //         float val = dense.data[i * n + j];
    //         if (val != 0.0f)
    //         {
    //             csr.Ai.push_back(j);
    //             csr.Ax.push_back(val);
    //         }
    //     }
    // }

    // csr.Ap[m] = csr.Ax.size();       // end of last row
    // return csr;
//}




}

#endif //PROJECT_DENSE_MATMUL_DEF_H
