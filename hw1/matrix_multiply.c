/**
 * Copyright (c) 2012 MIT License by 6.172 Staff
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 **/

#include "./matrix_multiply.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <math.h>
#include <string.h>

#include "./tbassert.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

// Allocates a row-by-cols matrix and returns it
matrix* make_matrix(int rows, int cols) {
  matrix* new_matrix = malloc(sizeof(matrix));

  // Set the number of rows and columns
  new_matrix->rows = rows;
  new_matrix->cols = cols;

  // Allocate a buffer big enough to hold the matrix.
  new_matrix->values = (int**)malloc(sizeof(int*) * rows);
  for (int i = 0; i < rows; i++) {
    new_matrix->values[i] = (int*)malloc(sizeof(int) * cols);
  }

  return new_matrix;
}

// Frees an allocated matrix
void free_matrix(matrix* m) {
  for (int i = 0; i < m->rows; i++) {
    free(m->values[i]);
  }
  free(m->values);
  free(m);
}

// Print matrix
void print_matrix(const matrix* m) {
  printf("------------\n");
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf("  %3d  ", m->values[i][j]);
    }
    printf("\n");
  }
  printf("------------\n");
}


// Multiply matrix A*B, store result in C.
int matrix_multiply_run(const matrix* A, const matrix* B, matrix* C) {
  tbassert(A->cols == B->rows,
           "A->cols = %d, B->rows = %d\n", A->cols, B->rows);
  tbassert(A->rows == C->rows,
           "A->rows = %d, C->rows = %d\n", A->rows, C->rows);
  tbassert(B->cols == C->cols,
           "B->cols = %d, C->cols = %d\n", B->cols, C->cols);

  for (int i = 0; i < C->rows; i++) {
    for (int j = 0; j < C->cols; j++) {
      C->values[i][j] = 0;
    }
  }

  // original
  // for (int i = 0; i < A->rows; i++) {
  //   for (int k = 0; k < A->cols; k++) {
  //     for (int j = 0; j < B->cols; j++) {
  //       C->values[i][j] += A->values[i][k] * B->values[k][j];
  //     }
  //   }
  // }

  // Tiling
  int s = 128; // you can change it to 4 / 8 / 16 / 32 / 64.
  for (int i1 = 0; i1 < A->rows; i1 += s) {
    int iStart = min(i1 + s, A->rows);
    for (int k1 = 0; k1 < A->cols; k1 += s) {
      int kStart = min(k1 + s, A->cols);
      for (int j1 = 0; j1 < B->cols; j1 += s) {
        int jStart = min(j1 + s, B->cols);
        for (int i = i1; i < iStart; i++) {
          for (int k = k1; k < kStart; k++) {
            for (int j = j1; j < jStart; j++) {
              C->values[i][j] += A->values[i][k] * B->values[k][j];
            }
          }
        }
      }
    }
  }

  return 0;
}

// Multiply matrix A*B, store result in C.
int matrix_multiply_verify(const matrix* A, const matrix* B, matrix* C) {
  for (int i = 0; i < A->rows; i++) {
    for (int k = 0; k < A->cols; k++) {
      for (int j = 0; j < B->cols; j++) {
        C->values[i][j] += A->values[i][k] * B->values[k][j];
      }
    }
  }

  return 0;
}
