#include <stdio.h>
#include <getopt.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/time.h>

#define NUM_TIMERS 8

#define EPS 1e-3
#define MS 1e+3

#define ABS_ERR(x, y) (fabsf((x) - (y)) > EPS)
#define REL_ERR(x, y) ((y) == 0 || fabsf((x) - (y)) / (y) > EPS)

#define GROUP_TILE_SIZE 64
#define ITEM_TILE_ROW_SIZE 4
#define ITEM_TILE_COL_SIZE 8
#define ROW_GAP (GROUP_TILE_SIZE / ITEM_TILE_ROW_SIZE)
#define COL_GAP (GROUP_TILE_SIZE / ITEM_TILE_COL_SIZE)


size_t sizeA, sizeB, sizeC;
float *a_global, *b_global, *c_global;

__global__ void kernel(const float *a, const float *b, float *c, const int N, const int M) {

	const int local_col = threadIdx.x;
	const int local_row = threadIdx.y;
	const int global_col = blockIdx.x * GROUP_TILE_SIZE + local_col;
	const int global_row = blockIdx.y * GROUP_TILE_SIZE + local_row;

	__shared__ float local_a[GROUP_TILE_SIZE][GROUP_TILE_SIZE];
	__shared__ float local_b[GROUP_TILE_SIZE][GROUP_TILE_SIZE];

	float acc[ITEM_TILE_ROW_SIZE][ITEM_TILE_COL_SIZE] = {0.0f,};

	const int tile_num = M / GROUP_TILE_SIZE;

	for(int i = 0; i < tile_num; i++) {
		for(int j = 0; j < ITEM_TILE_ROW_SIZE; j++) {
			const int local_tile_row = ROW_GAP * j + local_row;
			const int global_tile_row = ROW_GAP * j + global_row;
			
			for(int k = 0; k < ITEM_TILE_COL_SIZE; k++) {
				const int local_tile_col = COL_GAP * k + local_col;
				const int global_tile_col = COL_GAP * k + global_col;

				local_a[local_tile_col][local_tile_row] = a[global_tile_row * M + (GROUP_TILE_SIZE * i + local_tile_col)];
				local_b[local_tile_row][local_tile_col] = b[(GROUP_TILE_SIZE * i + local_tile_row) * N + global_tile_col];
			}
		}

		__syncthreads();

		for(int j = 0; j < GROUP_TILE_SIZE; j++)
			for(int k = 0; k < ITEM_TILE_ROW_SIZE; k++)
				for(int l = 0; l < ITEM_TILE_COL_SIZE; l++)
					acc[k][l] += local_a[j][ROW_GAP * k + local_row] * local_b[j][COL_GAP * l + local_col];

		__syncthreads();
	}
	
	for(int i = 0; i < ITEM_TILE_ROW_SIZE; i++) {
		const int global_tile_row = ROW_GAP * i + global_row;
		
		for(int j = 0; j < ITEM_TILE_COL_SIZE; j++) {
			const int global_tile_col = COL_GAP * j + global_col;

			c[global_tile_row * N + global_tile_col] = acc[i][j];
		}
	}
}

void printError(cudaError_t error) {
	if(error != cudaSuccess) {
		printf("\n%s\n", cudaGetErrorName(error));
		printf("\n%s\n", cudaGetErrorString(error));
		exit(0);
	}
}

void mat_mul(float *a, float *b, float *c, int N, int M) {

	printError(cudaMemcpy(a_global, a, sizeA, cudaMemcpyHostToDevice));
	printError(cudaMemcpy(b_global, b, sizeB, cudaMemcpyHostToDevice));

	dim3 dimBlock(GROUP_TILE_SIZE / ITEM_TILE_COL_SIZE, GROUP_TILE_SIZE / ITEM_TILE_ROW_SIZE);
	dim3 dimGrid(N / GROUP_TILE_SIZE, N / GROUP_TILE_SIZE);
	kernel<<<dimGrid, dimBlock>>>(a_global, b_global, c_global, N, M);

	printError(cudaMemcpy(c, c_global, sizeC, cudaMemcpyDeviceToHost));

	printError(cudaFree(a_global));
	printError(cudaFree(b_global));
	printError(cudaFree(c_global));
}

void setup(int N, int M) {
	
	sizeA = sizeB = N * M * sizeof(float);
	sizeC = N * N * sizeof(float);

	printError(cudaMalloc(&a_global, sizeA));
	printError(cudaMalloc(&b_global, sizeB));
	printError(cudaMalloc(&c_global, sizeC));
}

/////////////////////////////////////////////////////////////////////////////////
// main routine
/////////////////////////////////////////////////////////////////////////////////

static double start_time[NUM_TIMERS];

static double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

void timer_start(int i) {
  start_time[i] = get_time();
}

double timer_stop(int i) {
  return get_time() - start_time[i];
}

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
  setup(N, M);
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
