/**
 * Fernuniversität Hagen, Course 1727
 * Implementation of help functions for programming exercises with matrices.
 *
 * Author: Elena Tikhonova;
 * Date. October 2022
 */
#ifndef MATRIX_FUNCTIONS_H
#define MATRIX_FUNCTIONS_H
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <fstream>
#define PRINT_TOTAL_WIDTH "5"

#include <mpi.h>

double matA_small[16] = {8.0f, 2.0f, 8.0f, 2.0f, 9.0f, 2.0f, 9.0f, 9.0f,
                         6.0f, 1.0f, 4.0f, 4.0f, 10.0f, 9.0f, 4.0f, 4.0f};

double matB_small[16] = {3.0f, 0.0f, 10.0f, 2.0f, 7.0f, 4.0f, 1.0f, 3.0f,
                         1.0f, 1.0f, 0.0f, 7.0f, 2.0f, 2.0f, 4.0f, 10.0f};

void fill_buffer(double *send_buffer, size_t size)
{

  FILE *fileptr;
  char *buffer;

  size = 198180864;

  fileptr =
      fopen("/p/scratch/icei-hbp-2022-0013/vogel6/fp64/obs_spitzer.trace", "rb");
  // fopen("/home/marco/entwicklung/uni/cannons_algorithm_cuda/num_brain.trace",
  // "rb");

  buffer = (char *)malloc(size * sizeof(char));
  fread(buffer, size, sizeof(char), fileptr);
  fclose(fileptr);

  double *test = (double *)buffer;

  memcpy(send_buffer, buffer, size);
  free(buffer);
}

void readFileData(double **buffer)
{
  char *memblock;
  std::streampos size;
  std::ifstream file("/p/scratch/icei-hbp-2022-0013/vogel6/fp64/obs_spitzer.trace", std::ios::in | std::ios::binary | std::ios::ate);
  if (file.is_open())
  {
    size = file.tellg();

    //std::cout << "size=" << size << "\n";

    memblock = new char[size];
    file.seekg(0, std::ios::beg);
    file.read(memblock, size);
    file.close();

    double *double_values = (double *)memblock; // reinterpret as doubles

    *buffer = double_values;
  }
  else
  {
    std::cout << "Error opening file" << std::endl;
  }
}

/**
    Initialisiert eine Matrix mit randomisierten Zahlen aus dem Intervall [0..9]
   oder mit Nullen.
    @param A_out, Verweis auf die Ausgabedaten.
    @param n,m die Anzahl Zeilen bzw. Spalten in der Ausgabematrix.
    @param zeroM zur Initialisierung mit Nullen.
*/
void init_matrix(double **A_out, size_t n, size_t m, bool zeroM, int flag)
{
  double m_size = n * m;
  size_t bytes = m_size * sizeof(double);
  *A_out = (double *)malloc(bytes);

  double *file_data = NULL;

  file_data = (double *)malloc(bytes);
  // fill_buffer(file_data, bytes);
  readFileData(&file_data);


  for (int i = 0; i < m_size; i++)
  {
    if (zeroM)
    {
      (*A_out)[i] = 0;
    }
    else if (flag == 0)
    {
      (*A_out)[i] = file_data[i % 24772608]; // 34874483];  matA_small[i];
    }
    else
    {
      (*A_out)[i] = file_data[i % 24772608]; // 34874483]; matB_small[i];
    }
  }

  // for (int i = 0; i < m_size; i++) {
  //   if (zeroM) {
  //     (*A_out)[i] = 0;
  //   } else if (flag == 0) {
  //     (*A_out)[i] = matA_small[i % 16];
  //   } else {
  //     (*A_out)[i] = matB_small[i % 16];
  //   }
  // }

  free(file_data);
};

/**
    Erzeugt eine Kopie der Eingabematrix A_in.
    @param A_in, Verweis auf die Eingabematrix.
    @param A_out, Verweis auf die Ausgabematrix.
    @param n,m die Anzahl Zeilen bzw. Spalten in der Eingabematrix.
*/

void copy_matrix(const double *A_in, double **A_out, int n, int m)
{
  const size_t bytes = n * m * sizeof(double);
  *A_out = (double *)malloc(bytes);
  memcpy((void *)*A_out, (const void *)A_in, bytes);
};

/**
    Gibt die Eingabematrix in die Konsole in der Form

    |a11 a12 ... a1n|
    ...
    |an1 an2 ... ann|

    aus.
    @param A, Verweis auf die Eingabematrix.
    @param n die Anzahl Zeilen in der Eingabematrix.
    @param m die Anzahl Spalten in der Eingabematrix.
*/
void print_matrix(double *A, int n, int m)
{
  for (int i = 0; i < n; i++)
  {
    printf("|");
    for (int j = 0; j < m; j++)
    {
      printf("%" PRINT_TOTAL_WIDTH ".2f ", A[i * m + j]);
    }
    printf("|\n");
  }
};

/**
    Gibt die Eingabematrix in die Konsole in der Form

    |a11 a12 ... a1n | b1 |
    ...
    |an1 an2 ... ann | bn |

    aus.

    @param A, Verweis auf die Eingabematrix.
    @param n die Anzahl Zeilen in der Eingabematrix.
    @param m die Anzahl Spalten in der Eingabematrix.
*/
void print_matrix_vector(double *A, int n, int m)
{
  // if (n <= 10){
  for (int i = 0; i < n; i++)
  {
    printf("| ");
    for (int j = 0; j < m - 1; j++)
    {
      printf("%" PRINT_TOTAL_WIDTH ".2f ", A[i * m + j]);
    }
    // der Vektor als eine getrennte letzte Spalte
    printf("| | %" PRINT_TOTAL_WIDTH ".2f |\n", A[i * m + m - 1]);
  }
  //}else{
  //    printf("\nMatrix der Größe %d > 10 kann nicht angezeigt werden!\n", n);
  //}
}

#endif
