#include <assert.h>
#include <cuda.h>
#include <cudaMatrixMultiply.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Thread block size: BLOCK_SIZE * BLOCK_SIZE
#define BLOCK_SIZE 32

__global__ void dgemm_gpu_simple(const double *a, const double *b, double *c,
                                 const int n) {

  // Allocate shared memory for the two blocks aSub and bSub.
  // Use two-dimensional matrices of size BLOCK_SIZE * BLOCK_SIZE
  __shared__ double aSub[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double bSub[BLOCK_SIZE][BLOCK_SIZE];

  const int Bx_offset = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int Ay_offset = blockIdx.y * BLOCK_SIZE + threadIdx.y;
  double tmp = 0;
  /* Go */
  for (int blocks = 0; blocks < gridDim.x; blocks += 1) {
    int Ax_offset = threadIdx.x + blocks * BLOCK_SIZE;
    int By_offset = threadIdx.y + blocks * BLOCK_SIZE;

    if (Ax_offset < n && Ay_offset < n)
      aSub[threadIdx.y][threadIdx.x] = a[Ax_offset + Ay_offset * n];
    else
      aSub[threadIdx.y][threadIdx.x] = 0;
    if (Bx_offset < n && By_offset < n)
      bSub[threadIdx.y][threadIdx.x] = b[Bx_offset + By_offset * n];
    else
      bSub[threadIdx.y][threadIdx.x] = 0;

    __syncthreads(); // Make sure that all threads had time to read the sub
                     // matrix.

    for (int i = 0; i < BLOCK_SIZE; i++)
      tmp += aSub[threadIdx.y][i] * bSub[i][threadIdx.x];

    __syncthreads();
  }
  if ((Bx_offset < n) && (Ay_offset < n))

    c[Bx_offset + n * Ay_offset] += tmp;
}

// get compute performance
float getGflops(int width, float time) {

  float gf = (1.0e-6 * width * width * width / time);

  return gf;
}

void multiplyMatrixCuda(double *matA, double *matB, double *matC, int n) {

  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  dgemm_gpu_simple<<<dimGrid, dimBlock>>>(matA, matB, matC, n);

  cudaDeviceSynchronize();

}
