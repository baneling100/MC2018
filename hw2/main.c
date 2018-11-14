#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "timer.h"

#define EPS 1e-3
#define MS 1e+3

#define ABS_ERR(x, y) (fabsf((x) - (y)) > EPS)
#define REL_ERR(x, y) ((y) == 0 || fabsf((x) - (y)) / (y) > EPS)

/* mul.c */
void mat_mul(float *a, float *b, float *c, int N, int M);

static bool print_matrix = false;
static bool validation = false;

static void check_mat_mul(float *a, float *b, float *c, int N, int M) {
  bool is_valid = true;

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float ans = 0;
      for (int k = 0; k < M; ++k) {
        ans += a[i * M + k] * b[k * N + j];
      }

      float res = c[i * N + j];
      if (ABS_ERR(res, ans) && REL_ERR(res, ans)) {
        printf("c[%d][%d] : answer = %f, result = %f\n",
               i, j, ans, res);
        is_valid = false;
      }
    }
  }

  if (is_valid) {
    printf("Validation:\tSUCCESS\n");
  }
  else {
    printf("Validation:\tFAILED\n");
  }
}

static void rand_mat(float **m, size_t R, size_t C) {
  if (m == NULL) {
    printf("Unable to allocate memory for matrix.\n");
    exit(EXIT_FAILURE);
  }

  *m = (float *) malloc(sizeof(float) * R * C);

  if (*m == NULL) {
    printf("Failed to allocate memory for matrix.\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < R; i++) { 
    for (int j = 0; j < C; j++) {
      (*m)[i * C + j] = (float) rand() / RAND_MAX - 0.5;
    }
  }
}

static void zero_mat(float **m, size_t R, size_t C) {
  if (m == NULL) {
    printf("Unable to allocate memory for matrix.\n");
    exit(EXIT_FAILURE);
  }

  *m = (float *) malloc(sizeof(float) * R * C);

  if (*m == NULL) {
    printf("Failed to allocate memory for matrix.\n");
    exit(EXIT_FAILURE);
  }

  memset(*m, 0, sizeof(float) * R * C);
}

static void print_mat(float *m, size_t R, size_t C) {
  for (int i = 0; i < R; i++) { 
    for (int j = 0; j < C; j++) {
      printf("%+.3f ", m[i * C + j]);
    }
    printf("\n");
  }
}

static void print_help(const char* prog_name) {
  printf(" Usage: %s NDIM MDIM [-pvh]\n", prog_name);
  printf(" OPTIONS\n");
  printf("    -p : print matrix.\n");
  printf("    -v : validate matrix multiplication.\n");
  printf("    -h : print this page.\n");
}

static void parse_opt(int argc, char *argv[]) {
  int opt;
  while ((opt = getopt(argc, argv, "pvh")) > 0) {
    switch(opt) {
      case 'p':
        print_matrix = true;
        break;
      case 'v':
        validation = true;
        break;
      case 'h':
      default:
        print_help(argv[0]);
        exit(EXIT_SUCCESS);
    }
  }
}

int main(int argc, char *argv[]) {
  //===============================================================
  // Command line parsing
  //===============================================================
  if (argc < 3) {
    print_help(argv[0]);
    exit(EXIT_FAILURE);
  }

  int N = atoi(argv[1]);
  int M = atoi(argv[2]);

  if (N < 1 || M < 1) {
    print_help(argv[0]);
    exit(EXIT_FAILURE);
  }

  parse_opt(argc, argv);

  printf("\nProblem size:\t%d x %d x %d\n\n", N, N, M);

  //===============================================================
  // Initialization
  //===============================================================
  float *a = NULL;
  float *b = NULL;
  float *c = NULL;

  printf(" Initializing ...\t"); fflush(stdout);
  rand_mat(&a, N, M);
  rand_mat(&b, M, N);
  zero_mat(&c, N, N);
  printf("done!\n");

  //===============================================================
  // Caculation
  //===============================================================
  printf(" Calculating ...\t"); fflush(stdout);
  timer_start(0);
  mat_mul(a, b, c, N, M);
  double elapsed_time = timer_stop(0);
  printf("done!\n");

  //===============================================================
  // Print results and Validation
  //===============================================================
  if (print_matrix) {
    printf("MATRIX A:\n"); print_mat(a, N, M);
    printf("MATRIX B:\n"); print_mat(b, M, N);
    printf("MATRIX C:\n"); print_mat(c, N, N);
  }
  if (validation) {
    printf(" Validation on.\n\n");
    check_mat_mul(a, b, c, N, M);
  }
  else {
    printf(" Validation off.\n\n");
  }

  printf("Elapsed time:\t%.3f sec (%.3f ms)\n\n",
         elapsed_time, elapsed_time * MS);

  return 0;
}
