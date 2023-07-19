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

  // dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 gridDim(ceil(n / blockDim.x), ceil(n / blockDim.y));

  double *subMatA = NULL;

  dim3 dimGrid((n + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

  // printf("gridDim.x:%d blockDim.x:%d %d\n", gridDim.x, blockDim.x, n);

  dgemm_gpu_simple<<<dimGrid, dimBlock>>>(matA, matB, matC, n);
  // gpu_square_matrix_mult<<<dimGrid,dimBlock>>>(*matA, *matB, *matC, n);
  //  matrixMultiplicationKernel<<<1,1>>>(*matA, *matB, *matC, n);
  // gpu_matrix_mult<<<dimGrid,dimBlock >>>(*matA, *matB, *matC, n, n, n);

  cudaDeviceSynchronize();
  // subMatA = (double *)malloc(4 * sizeof(double));
  // cudaMemcpy(subMatA, *matB, 4 * sizeof(double), cudaMemcpyDeviceToHost);

  // printf("[1]:%f %f\n%f %f\n", subMatA[0], subMatA[1], subMatA[2],
  // subMatA[3]);

  free(subMatA);
}

// int main(int argc, const char **argv) {

//   int n = 4096; // dimension of square matrices
//   double *d_a, *d_b, *d_c;
//   double *h_a, *h_b, *h_c;
//   int row, col;
//   double absError, maxAbsError = 0.0, sumAbsError = 0.0;
//   size_t size;
//   float time;
//   cudaEvent_t start, stop;

//   cudaSetDevice(1);
//   cudaEventCreate(&start);
//   cudaEventCreate(&stop);

//   if (argc > 1) {
//     n = atoi(argv[1]);
//   }

//   size = (n * n) * sizeof(double);

//   cudaMalloc(&d_a, size);
//   cudaMalloc(&d_b, size);
//   cudaMalloc(&d_c, size);

//   cudaMallocHost(&h_a, size);
//   cudaMallocHost(&h_b, size);
//   cudaMallocHost(&h_c, size);

// #pragma omp parallel for
//   for (row = 0; row < n; row++) {
//     for (col = 0; col < n; col++) {
//       h_a[row * n + col] = (row == col) ? 1.0 : 0.0;
//       h_b[row * n + col] = row * n + col;
//     }
//   }

//   // Execute matrix multiplication (on device and on host for reference
//   dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
//   dim3 gridDim(ceil(n / blockDim.x), ceil(n / blockDim.y));

//   cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
//   double runs[10];
//   int i;
//   dgemm_gpu_simple<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
//   cudaDeviceSynchronize();
//   for (i = 0; i < 10; i++) {

//     cudaEventRecord(start, 0);
//     multiplyMatrixCuda(d_a, d_b, d_c, n);
//     //dgemm_gpu_simple<<<gridDim, blockDim>>>(d_a, d_b, d_c, n);
//     cudaEventRecord(stop, 0);

//     cudaEventSynchronize(stop);

//     cudaEventElapsedTime(&time, start, stop);
//     runs[i] = time;
//   }
//   double total_time = 0.0;
//   for (i = 0; i < 10; i++) {
//     total_time += runs[i];
//   }
//   total_time /= 10;

//   cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

//   for (row = 0; row < n; ++row) {
//     for (col = 0; col < n; ++col) {

//       absError = fabs(h_c[row * n + col] - h_b[row * n + col]);
//       sumAbsError += absError;

//       if (absError > maxAbsError)
//         maxAbsError = absError;
//     }
//   }
//   // cudaEventElapsedTime(&time, start, stop);

//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);

//   // Free memory on host
//   cudaFree(d_a);
//   cudaFree(d_b);
//   cudaFree(d_c);
//   printf("%d\t%f\n", n, total_time);

//   printf("This corresponds to: %4.4f GFLOPS\n\n", getGflops(n, time));

//   return 0;
// }
