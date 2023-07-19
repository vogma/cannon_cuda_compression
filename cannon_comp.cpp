#include "matrix_functions.h"

#include "cuda_runtime.h"
#include <cudaMatrixMultiply.h>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <ndzip_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <time.h>

// checking whether the current process is the root process
#define IS_ROOT (rank == 0)

#define PRINTSIZELIMIT 11

// references to the local matrices
double *subMatA = NULL, *subMatB = NULL, *subMatC = NULL;

// references to the input and output matrices
double *matA = NULL, *matB = NULL;
double *matC = NULL;

double *matCheck = NULL;

// references to the local matrices in GPU memory
double *d_subMatA = NULL;
double *d_subMatB = NULL;
double *d_subMatC = NULL;

double *subMatCheck = NULL;

// References to the input and output matrices on the GPU.
double *d_matA = NULL;
double *d_matB = NULL;
double *d_matC = NULL;

u_int64_t *d_compress_buffer_a = NULL;
u_int64_t *d_compress_buffer_b = NULL;

double *d_receive_buffer_a = NULL;
double *d_receive_buffer_b = NULL;

NDZIP_API *NDZIP_API_instance = NULL;

double *allocateSubmatrix(double *subMat, double bytes, int rank) {
  if (!(subMat = (double *)malloc(bytes))) {
    fprintf(
        stderr,
        "Prozessor %d: Speicherzuweisung für %d bytes ist fehlgeschlagen.\n",
        rank, bytes);
    if (IS_ROOT) {
      free(matA);
      free(matB);
      free(matC);
      free(matCheck);
    }
    free(subMatA);
    free(subMatB);
    free(subMatC);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }
  return subMat;
}

bool isSquare(int x) {
  const float sqr = sqrtf(x);
  return (floor(sqr) == ceil(sqr));
}

void MatrixMatrixMultiplyCuda(int n, double *d_a, double *d_b, double *d_c,
                              MPI_Comm comm) {
  int i;
  int nlocal;
  int npes, dims[2], periods[2];
  int myrank, my2drank, mycoords[2];
  int uprank, downrank, leftrank, rightrank;
  int shiftsource, shiftdest;
  int procs_per_dim;

  double t_start_timing, t_end_timing, runtime_timing0;

  double start_timing_compute, end_timing_compute, start_timing_comm,
      end_timing_comm;

  double start_timing_compression, end_timing_compression;
  double start_timing_decompression, end_timing_decompression;

  MPI_Status status;
  //
  MPI_Comm comm_2d;

  /*Step 1: Collection of information about the MPI specified communicator
    ================================================================================*/
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &myrank);

  /*Step 2: Cartesian topology setup
    ===================================================*/
  procs_per_dim = (int)sqrt(npes);

  /*Schritt 2: Einrichtung einer Kartesischen Topologie
    ===================================================*/
  dims[0] = dims[1] = sqrt(npes);
  periods[0] = periods[1] = 1;
  MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);

  /*Determination of the rank and coordinates with respect to the new topology*/
  MPI_Comm_rank(comm_2d, &my2drank);
  MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

  /*Step 3: Initial alignment
   =================================*/
  MPI_Cart_shift(comm_2d, 1, -1, &rightrank,
                 &leftrank);                          // i. e. dim[1] is X-coord
  MPI_Cart_shift(comm_2d, 0, -1, &downrank, &uprank); // i. e. dim[0] is Y-coord

  /* size of local matrix*/
  nlocal = n / dims[0];

  MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest);

  MPI_Sendrecv_replace(d_a, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
                       shiftsource, 1, comm_2d, &status);

  /* Initial alignment for Matrix B */
  MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest);

  MPI_Sendrecv_replace(d_b, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
                       shiftsource, 1, comm_2d, &status);

  int compressed_length_a;
  int compressed_length_b;

  int recv_compressed_length_a;
  int recv_compressed_length_b;

  for (i = 0; i < dims[0]; i++) {
    
    /*local computation*/
    start_timing_compute = MPI_Wtime();
    multiplyMatrixCuda(d_a, d_b, d_c, nlocal);
    end_timing_compute = MPI_Wtime();

    start_timing_compression = MPI_Wtime();

    compressed_length_a = NDZIP_API_instance->compress_buffer(
        d_a, nlocal * nlocal * sizeof(double), d_compress_buffer_a);

    compressed_length_b = NDZIP_API_instance->compress_buffer(
        d_b, nlocal * nlocal * sizeof(double), d_compress_buffer_b);

    end_timing_compression = MPI_Wtime();

    start_timing_comm = MPI_Wtime();

    //-----------------------shift a left----------------------------//
    if (my2drank != 0 && (my2drank % procs_per_dim != 0)) {

      MPI_Recv(&recv_compressed_length_a, 1, MPI_INT, rightrank, 0, comm_2d,
               &status);

      MPI_Recv(d_receive_buffer_a, recv_compressed_length_a, MPI_DOUBLE,
               rightrank, 0, comm_2d, &status);
    }

    MPI_Send(&compressed_length_a, 1, MPI_INT, leftrank, 0, comm_2d);
    MPI_Send(d_compress_buffer_a, compressed_length_a, MPI_DOUBLE, leftrank, 0,
             comm_2d);

    if (my2drank == 0 || (my2drank % procs_per_dim == 0)) {
      MPI_Recv(&recv_compressed_length_a, 1, MPI_INT, rightrank, 0, comm_2d,
               &status);
      MPI_Recv(d_receive_buffer_a, recv_compressed_length_a, MPI_DOUBLE,
               rightrank, 0, comm_2d, &status);
    }

    //-------------------------------------------------------------------//

    //--------------------------shift b up-------------------------------//
    if (my2drank >= procs_per_dim) {

      MPI_Recv(&recv_compressed_length_b, 1, MPI_INT, downrank, 0, comm_2d,
               &status);

      MPI_Recv(d_receive_buffer_b, recv_compressed_length_b, MPI_DOUBLE,
               downrank, 0, comm_2d, &status);
    }

    MPI_Send(&compressed_length_b, 1, MPI_INT, uprank, 0, comm_2d);

    MPI_Send(d_compress_buffer_b, compressed_length_b, MPI_DOUBLE, uprank, 0,
             comm_2d);

    if (my2drank < procs_per_dim) {
      MPI_Recv(&recv_compressed_length_b, 1, MPI_INT, downrank, 0, comm_2d,
               &status);
      MPI_Recv(d_receive_buffer_b, recv_compressed_length_b, MPI_DOUBLE,
               downrank, 0, comm_2d, &status);
    }

    end_timing_comm = MPI_Wtime();

    start_timing_decompression = MPI_Wtime();

    d_a = NDZIP_API_instance->decompress_buffer(
        (long unsigned int *)d_receive_buffer_a, d_a,
        nlocal * nlocal * sizeof(double));

    d_b = NDZIP_API_instance->decompress_buffer(
        (long unsigned int *)d_receive_buffer_b, d_b,
        nlocal * nlocal * sizeof(double));

    end_timing_decompression = MPI_Wtime();

    std::cout << "rank " << my2drank << " run " << i
              << " matrix size: " << nlocal << " computation time: "
              << (end_timing_compute - start_timing_compute) * 1000
              << " compression time: "
              << ((end_timing_compression - start_timing_compression) +
                  (end_timing_decompression - start_timing_decompression)) *
                     1000
              << " communication time: "
              << (end_timing_comm - start_timing_comm) * 1000
              << " total runtime: "
              << (end_timing_decompression - start_timing_compute) * 1000
              << std::endl;

    //-----------------------------------------------------------------------
  }

  MPI_Cart_shift(comm_2d, 1, +mycoords[0], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace(d_a, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
                       shiftsource, 1, comm_2d, &status);

  MPI_Cart_shift(comm_2d, 0, +mycoords[1], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace(d_b, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
                       shiftsource, 1, comm_2d, &status);

  MPI_Comm_free(&comm_2d);
}

int main(int argc, char *argv[]) {

  int n = std::stoi(argv[1]);

  NDZIP_API_instance = new NDZIP_API();

  double t_start_timing0, t_end_timing0, runtime_timing0;
  t_start_timing0 = MPI_Wtime();

  int size, rank;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (!isSquare(size)) {
    printf("Fehlermeldung: Das Program soll mit eine quadratische Anzahl "
           "Prozessoren gestartet werden!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Number of matrix blocks in the X-direction and the Y-direction
  int tilesX;
  int tilesY;

  // Number of elements per block in the X direction
  int tileSizeX;
  int tileSizeY;

  // number of matrix blocks
  tilesX = sqrtf(size);
  tilesY = tilesX;

  // size of matrix block
  tileSizeX = n / tilesX;
  tileSizeY = n / tilesY;
  double t_start_init0, t_end_init0, runtime_init0;

  t_start_init0 = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (IS_ROOT) {

    init_matrix(&matA, n, n, false, argv[2]);
    init_matrix(&matB, n, n, false, argv[2]);
    init_matrix(&matC, n, n, true, argv[2]);
    init_matrix(&matCheck, n, n, true, argv[2]);

    cudaMalloc(&d_matA, (n * n * sizeof(double)));
    cudaMalloc(&d_matB, (n * n * sizeof(double)));
    cudaMalloc(&d_matC, (n * n * sizeof(double)));

    cudaMemcpy(d_matA, matA, (n * n * sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, (n * n * sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matC, matC, (n * n * sizeof(double)), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
  }
  if (size > 1) {

    subMatA = allocateSubmatrix(
        subMatA, (double)tileSizeX * tileSizeX * sizeof(double), rank);
    subMatB = allocateSubmatrix(
        subMatB, (double)tileSizeX * tileSizeX * sizeof(double), rank);
    subMatC = allocateSubmatrix(
        subMatC, (double)tileSizeX * tileSizeX * sizeof(double), rank);
    subMatCheck = allocateSubmatrix(
        subMatCheck, tileSizeX * tileSizeX * sizeof(double), rank);

    cudaMalloc(&d_compress_buffer_a,
               tileSizeX * tileSizeX * sizeof(double) * 2);
    cudaMalloc(&d_compress_buffer_b,
               tileSizeX * tileSizeX * sizeof(double) * 2);

    cudaMalloc(&d_receive_buffer_a, tileSizeX * tileSizeX * sizeof(double) * 2);
    cudaMalloc(&d_receive_buffer_b, tileSizeX * tileSizeX * sizeof(double) * 2);

    cudaMalloc(&d_subMatA, tileSizeX * tileSizeX * sizeof(double) * 2);
    cudaMalloc(&d_subMatB, tileSizeX * tileSizeX * sizeof(double) * 2);
    cudaMalloc(&d_subMatC, tileSizeX * tileSizeX * sizeof(double) * 2);

    cudaDeviceSynchronize();

    // Distribution of matrices between GPU's
    MPI_Scatter(d_matA, tileSizeX * tileSizeX, MPI_DOUBLE, d_subMatA,
                tileSizeX * tileSizeX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(d_matB, tileSizeX * tileSizeX, MPI_DOUBLE, d_subMatB,
                tileSizeX * tileSizeX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    t_end_init0 = MPI_Wtime();

    runtime_init0 = t_end_init0 - t_start_init0;
  }

  double t_start_compute0, t_end_compute0, runtime_compute0;
  t_start_compute0 = MPI_Wtime();

  MatrixMatrixMultiplyCuda(tilesX * tileSizeX, d_subMatA, d_subMatB, d_subMatC,
                           MPI_COMM_WORLD);

  // Collection of results from all GPU's
  MPI_Gather(d_subMatC, tileSizeX * tileSizeX, MPI_DOUBLE, d_matC,
             tileSizeX * tileSizeX, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    t_end_compute0 = MPI_Wtime();

    cudaMemcpy(matCheck, d_matC, n * n * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    runtime_compute0 = t_end_compute0 - t_start_compute0;
  }

  if (IS_ROOT) {

    free(matA);
    free(matB);
    free(matC);
    free(matCheck);

    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);
  }
  if (size > 1) {
    free(subMatA);
    free(subMatCheck);
    free(subMatB);
    free(subMatC);

    cudaFree(d_receive_buffer_a);
    cudaFree(d_receive_buffer_b);
    cudaFree(d_compress_buffer_a);
    cudaFree(d_compress_buffer_b);
  }

  MPI_Finalize();
  if (rank == 0) {
    t_end_timing0 = MPI_Wtime();

    runtime_timing0 = t_end_timing0 - t_start_timing0;
  }

  return 0;
}
