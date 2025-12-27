// Created by SwiftWare Lab on 2025-09-25.
// Course: CE 4SP4 - High Performance Programming
// Copyright (c) 2025 SwiftWare Lab. All rights reserved.
//
// Distribution of this code is not permitted in any form
// without express written permission from SwiftWare Lab.

#include "spmv.h"
#include <immintrin.h> 

namespace swiftware::hpp {
  // void spmvCSR(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp)
  // {
  //   // TODO Implement SPMV: c = A * b + c
  //   // Loop over all rows
  //   for (int row = 0; row < m; row++)
  //   {
  //     float sum = 0.0f;

  //     // start and end of this row in CSR format
  //     int start = Ap[row];
  //     int end   = Ap[row + 1];
  //     // Go through all nonzero entries in this row
  //     for (int idx = start; idx < end; idx++)
  //     {
  //       int col = Ai[idx];   // column index of this value
  //       float val = Ax[idx]; // actual nonzero value
  //       sum += val * b[col];
  //     }

  //     c[row] = sum;
      
  //   }

  // }
  void spmvCSR(int m, int n,const int *Ap, const int *Ai, const float *Ax,const float *b, float *c,ScheduleParams Sp)
  {
    for (int row = 0; row < m; row++)
    {
      float sum = c[row];   // <-- accumulate, not overwrite

      int start = Ap[row];
      int end   = Ap[row + 1];

      for (int idx = start; idx < end; idx++)
      {
        int col   = Ai[idx];
        float val = Ax[idx];
        sum += val * b[col];
      }

      c[row] = sum;
    }
  }

  void spmvCSR_AVX(int m, int n,const int *Ap,const int *Ai,const float *Ax,const float *b,float *c,ScheduleParams Sp)
  {
    constexpr int VEC = 8;

    for (int row = 0; row < m; row++)
    {
      float sum = c[row];

      int start = Ap[row];
      int end   = Ap[row + 1];

      int idx = start;

      // Process 8 nonzeros at a time
      for (; idx + VEC <= end; idx += VEC)
      {
      
        __m256 val_vec  = _mm256_loadu_ps(&Ax[idx]);

        // Gather irregular b[col] elements (took my life to write this line)
        __m256 b_vec = _mm256_set_ps(b[Ai[idx+7]], b[Ai[idx+6]], b[Ai[idx+5]], b[Ai[idx+4]],b[Ai[idx+3]], b[Ai[idx+2]], b[Ai[idx+1]], b[Ai[idx+0]]);

        __m256 mul_vec = _mm256_mul_ps(val_vec, b_vec);

        float tmp[8];
        _mm256_storeu_ps(tmp, mul_vec);

        sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
      }

      // Scalar tail
      for (; idx < end; idx++)
      {
          int col   = Ai[idx];
          float val = Ax[idx];
          sum += val * b[col];
      }

      c[row] = sum;
    }
  }



}
