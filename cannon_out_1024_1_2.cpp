#include "matrix_functions.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
// #include
// </p/project/icei-hbp-2022-0013/vogel6/cannons_algorithm_cuda/cudaMatrixMultiply.h>
#include "cuda_runtime.h"
#include <boost/program_options.hpp>
#include <cudaMatrixMultiply.h>
#include <io/io.hh>
#include <iostream>
#include <ndzip/ndzip.hh>
#include <ndzip_api.h>
#include <string>

// Abkürzung für die Überprüfung, ob der aktuelle Prozess der Wurzelprozess ist
#define IS_ROOT (rank == 0)

#define PRINTSIZELIMIT 11
// #define DEBUG

// Verweise auf die lokalen Matrizen
double *subMatA = NULL, *subMatB = NULL, *subMatC = NULL;

// Verweise auf die Ein- und Ausgabematrizen (allokiert beim Wurzelprozess)
double *matA = NULL, *matB = NULL;
double *matC = NULL;

double *matCheck = NULL;

// Verweise auf die lokalen Matrizen im GPU Speicher
double *d_subMatA = NULL;
double *d_subMatB = NULL;
double *d_subMatC = NULL;

double *subMatCheck = NULL;

// Verweise auf die Ein- und Ausgabematrizen auf der GPU
double *d_matA = NULL;
double *d_matB = NULL;
double *d_matC = NULL;

u_int64_t *d_compress_buffer_a = NULL;
u_int64_t *d_compress_buffer_b = NULL;

double *d_receive_buffer_a = NULL;
double *d_receive_buffer_b = NULL;

NDZIP_API *NDZIP_API_instance = NULL;

/**
 * Reserviert Speicher für eine Submatrix
 */
double *allocateSubmatrix(double *subMat, double bytes, int rank) {
  if (!(subMat = (double *)malloc(bytes))) {
    fprintf(
        stderr,
        "Prozessor %d: Speicherzuweisung für %d bytes ist fehlgeschlagen.\n",
        rank, bytes);
    // Freigabe der bis jetzt allokierten Speicher
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

/** Implementiert die sequenzielle Version der Matrix-Matrix Multiplikation
 * @param n die Anzahl Elemente pro Matrixblock. n muss eine Quadratzahl sein.
 * @param *a, *b Verweise auf die Eingabematrizen
 * @param *c Verweis auf die Ausgabematrizen**/
void MatrixMultiply(int n, double *a, double *b, double *c) {
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      for (int k = 0; k < n; k++)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
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

  /*Schritt 1: Sammlung der Informationen über den von MPI vorgegebenen
    Kommunikator
    ================================================================================*/
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &myrank);

  // unsafe but npes is guaranteed to be square
  procs_per_dim = (int)sqrt(npes);

  /*Schritt 2: Einrichtung einer Kartesischen Topologie
    ===================================================*/
  dims[0] = dims[1] = sqrt(npes);
  periods[0] = periods[1] = 1;
  MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);

  /* Bestimmung des Ranges und der Koordinaten bezüglich der neuen Topologie */
  MPI_Comm_rank(comm_2d, &my2drank);
  MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

  /*Schritt 3: Initiale Verschiebung
   =================================*/
  /* Berechnung der Ränge der Komminikationspartners links, rechts, oben und
   * unten*/
  MPI_Cart_shift(comm_2d, 1, -1, &rightrank,
                 &leftrank);                          // i. e. dim[1] is X-coord
  MPI_Cart_shift(comm_2d, 0, -1, &downrank, &uprank); // i. e. dim[0] is Y-coord

  /* Bestimmung der Größe von lokalen Matrizen*/
  nlocal = n / dims[0];

  /* Durchführung der initialen Verschiebung für die Matrix A */
  MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest);
  // This is different to the solution from the book by GRAMA:
  // above we chose that dim[1] is X-coord (column)
  // Matrix A has to be shifted left according to the own Y-position

  MPI_Sendrecv_replace(d_a, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
                       shiftsource, 1, comm_2d, &status);

  /* Durchführung der initialen Verschiebung für die Matrix B */
  MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest);
  // This is different to the solution from the book by GRAMA:
  // above we chose that dim[0] is Y-coord (column)
  // Matrix B has to be shifted up according to the own X-position

  MPI_Sendrecv_replace(d_b, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
                       shiftsource, 1, comm_2d, &status);

  int compressed_length_a;
  int compressed_length_b;

  int recv_compressed_length_a;
  int recv_compressed_length_b;

  /* Hauptschleife */
  for (i = 0; i < dims[0]; i++) {

    /*Lokale Berechnungen
      -------------------*/

    start_timing_compute = MPI_Wtime();
    multiplyMatrixCuda(d_a, d_b, d_c, nlocal);
    end_timing_compute = MPI_Wtime();

    std::cout << "before compression " << nlocal * nlocal << std::endl;

    start_timing_compression = MPI_Wtime();

    compressed_length_a = NDZIP_API_instance->compress_buffer(
        d_a, nlocal * nlocal * sizeof(double), d_compress_buffer_a);

    compressed_length_b = NDZIP_API_instance->compress_buffer(
        d_b, nlocal * nlocal * sizeof(double), d_compress_buffer_b);

    end_timing_compression = MPI_Wtime();

    std::cout << "b after compression " << compressed_length_b << std::endl;

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

      std::cout << "Rank " << my2drank << " received " << compressed_length_a
                << " MPI_DOUBLE's from rank " << rightrank << std::endl;

      MPI_Recv(d_receive_buffer_b, recv_compressed_length_b, MPI_DOUBLE,
               downrank, 0, comm_2d, &status);
    }

    MPI_Send(&compressed_length_b, 1, MPI_INT, uprank, 0, comm_2d);
    std::cout << "Rank " << my2drank << " sending " << compressed_length_a
              << " MPI_DOUBLE's to rank " << leftrank << std::endl;

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

  /* Herstellung der initialen Verteilung von Matrizen A und B */
  MPI_Cart_shift(comm_2d, 1, +mycoords[0], &shiftsource, &shiftdest);
  // again, this is different to the solution from the book by GRAMA:
  MPI_Sendrecv_replace(d_a, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
                       shiftsource, 1, comm_2d, &status);

  MPI_Cart_shift(comm_2d, 0, +mycoords[1], &shiftsource, &shiftdest);
  // again, this is different to the solution from the book by GRAMA:
  MPI_Sendrecv_replace(d_b, nlocal * nlocal, MPI_DOUBLE, shiftdest, 1,
                       shiftsource, 1, comm_2d, &status);

  MPI_Comm_free(&comm_2d);
}

int main(int argc, char *argv[]) {

  FILE *out_file = fopen(argv[2], "w");

  if (out_file == NULL) {
    printf("Error! Could not open file");
    return -1;
  }

  int n = std::stoi(argv[1]);

  NDZIP_API_instance = new NDZIP_API();

  fprintf(out_file, "0\n");
  fprintf(out_file, "1 1024\n");
  // start_main
  double t_start_timing0, t_end_timing0, runtime_timing0;
  t_start_timing0 = MPI_Wtime();

  int size, rank;

  // size_t n = 16384;
  //  size_t n = 15376;
  //  size_t n = 11500;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (!isSquare(size)) {
    printf("Fehlermeldung: Das Program soll mit eine quadratische Anzahl "
           "Prozessoren gestartet werden!\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // Anzahl Matrixblöcke in die X-Richtung und die Y-Richtung
  int tilesX;
  int tilesY;

  // Anzahl Elemente pro Block in die X-Richtung
  int tileSizeX;
  int tileSizeY;

  // Bestimme die Anzahl der Matrixblöcke
  tilesX = sqrtf(size);
  tilesY = tilesX;

  // Bestimme die Größe eines Matrixblocks
  printf("tileSizeX: %d\n", n / tilesX);
  tileSizeX = n / tilesX;
  // TODO check if we really need it
  tileSizeY = n / tilesY;
  double t_start_init0, t_end_init0, runtime_init0;

  t_start_init0 = MPI_Wtime();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (IS_ROOT) {
    // Initialisierung

    init_matrix(&matA, n, n, false, 0);
    init_matrix(&matB, n, n, false, 1);
    init_matrix(&matC, n, n, true, 1);
    init_matrix(&matCheck, n, n, true, 1);

    // for (int i = 0; i <= 100; i++)
    // {
    //   if (i % 10 == 0)
    //   {
    //     std::cout << "\n";
    //   }
    //   std::cout << matA[i] << " ";
    // }

    cudaMalloc(&d_matA, (n * n * sizeof(double)));
    cudaMalloc(&d_matB, (n * n * sizeof(double)));
    cudaMalloc(&d_matC, (n * n * sizeof(double)));

    cudaMemcpy(d_matA, matA, (n * n * sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, (n * n * sizeof(double)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matC, matC, (n * n * sizeof(double)), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
  }
  if (size > 1) {
    // begin

    // Allokierung der lokalen Matrizen
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

    // Verteilung der Matrix A
    // MPI_Scatter(matA, tileSizeX * tileSizeX, MPI_DOUBLE, subMatA,
    //          tileSizeX * tileSizeX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Verteilung der Matrix B
    // MPI_Scatter(matB, tileSizeX * tileSizeX, MPI_DOUBLE, subMatB,
    //          tileSizeX * tileSizeX, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Verteilung der Matrizen zwischen den GPU's
    MPI_Scatter(d_matA, tileSizeX * tileSizeX, MPI_DOUBLE, d_subMatA,
                tileSizeX * tileSizeX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(d_matB, tileSizeX * tileSizeX, MPI_DOUBLE, d_subMatB,
                tileSizeX * tileSizeX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    t_end_init0 = MPI_Wtime();

    runtime_init0 = t_end_init0 - t_start_init0;
    fprintf(out_file, "init_0: %f\n", runtime_init0);
  }

  double t_start_compute0, t_end_compute0, runtime_compute0;
  t_start_compute0 = MPI_Wtime();

  if (size == 1) {
    MatrixMultiply(n, matA, matB, matC);
  } else {

    MatrixMatrixMultiplyCuda(tilesX * tileSizeX, d_subMatA, d_subMatB,
                             d_subMatC, MPI_COMM_WORLD);

    // Sammlung der Ergebnisse aller GPU's
    MPI_Gather(d_subMatC, tileSizeX * tileSizeX, MPI_DOUBLE, d_matC,
               tileSizeX * tileSizeX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    t_end_compute0 = MPI_Wtime();

    cudaMemcpy(matCheck, d_matC, n * n * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // print_matrix(matA, n, n);
    // printf("\n");
    // print_matrix(matB, n, n);
    // printf("\n\n");
    // printf("-----------------------\n");
    // print_matrix(matCheck, n, n);
    // printf("-----------------------\n\n\n");

    // for (int i = 0; i <= 100; i++) {
    //   if (i % 10 == 0) {
    //     std::cout << "\n";
    //   }
    //   std::cout << matCheck[i] << " ";
    // }

    runtime_compute0 = t_end_compute0 - t_start_compute0;
    fprintf(out_file, "compute_0: %f\n", runtime_compute0);
  }

  if (IS_ROOT) {
    // printf("Matrix C:\n");
    // printMatrix(tileSizeX, tileSizeY, tilesX, tilesY, matC);

    // printf("Zeit für die Initialisierung von Eingabedaten:     %12.6f
    // Sek.\n", tEndInput-tStartInput); printf("Zeit für die Datenverteilung:
    // %12.6f Sek.\n", tEndDistribution-tStartDistribution); printf("Zeit für
    // die Matrixmultiplikation:                 %12.6f Sek.\n",
    // tStartCollection-tEndDistribution); printf("Zeit für Sammlung der
    // Ergebnisse:                  %12.6f Sek.\n",
    // tEndCollection-tStartCollection);

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
    fprintf(out_file, "timing_0: %f\n", runtime_timing0);
  }

  return 0;
}
