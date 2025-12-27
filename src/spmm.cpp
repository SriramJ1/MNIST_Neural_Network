#include "spmm.h"
#include <immintrin.h>

namespace swiftware::hpp {

void spmmCSR(int m, int n, int k,const int *Ap,const int *Ai,const float *Ax,const float *B,float *C,ScheduleParams Sp)
{
    // A is m × k in CSR
    // B is k × n dense (row-major)
    // C is m × n dense (row-major)
    // required: C = A*B + C

    for (int row = 0; row < m; row++)
    {
      // pointer to start of row 'row' in C
      float *Crow = C + row * n;

      int start = Ap[row];
      int end   = Ap[row + 1];

      // go over all non-zero entries in row of A
      for (int idx = start; idx < end; idx++)
      {
        int colA   = Ai[idx];   // column in A, row in B
        float valA = Ax[idx];   // A[row, colA]

        // pointer to row 'colA' of B
        const float *Brow = B + colA * n;

        // C[row][j] += A[row][colA] * B[colA][j]
        for (int j = 0; j < n; j++)
        {
            Crow[j] += valA * Brow[j];
        }
      }
    }
}
void spmmCSR_AVX(int m, int n, int k,
                 const int *Ap,
                 const int *Ai,
                 const float *Ax,
                 const float *B,
                 float *C)
{
    const int VEC = 8;   // AVX processes 8 floats at a time

    for (int row = 0; row < m; row++)
    {
        float *Crow = C + row * n;

        int start = Ap[row];
        int end   = Ap[row + 1];

        // Loop over non-zero elements in CSR row
        for (int idx = start; idx < end; idx++)
        {
            int colA = Ai[idx];
            float a  = Ax[idx];

            const float *Brow = B + colA * n;

            // Broadcast A[row][colA] into AVX register
            __m256 a_vec = _mm256_set1_ps(a);

            int j = 0;

            // ---- Vector loop (8 floats per step) ----
            for (; j <= n - VEC; j += VEC)
            {
                __m256 c_vec = _mm256_loadu_ps(Crow + j);
                __m256 b_vec = _mm256_loadu_ps(Brow + j);

                // Crow[j..j+7] += a * Brow[j..j+7]
                __m256 res = _mm256_fmadd_ps(a_vec, b_vec, c_vec);

                _mm256_storeu_ps(Crow + j, res);
            }

            // ---- Scalar tail ----
            for (; j < n; j++)
            {
                Crow[j] += a * Brow[j];
            }
        }
    }
}

} // namespace swiftware::hpp
