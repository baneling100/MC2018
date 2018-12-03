#include "mpi.h"
#include <omp.h>
#define NUM_THREADS 32
//====================================================================
// a[N][M] x b[M][N] = c[N][N]
//====================================================================
void mat_mul(float *a, float *b, float *c, int N, int M) {

	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int local_N = N / size;

	MPI_Request request_a, request_b;
	MPI_Status status_a, status_b;

	MPI_Iscatter(a, local_N * M, MPI_FLOAT, a, local_N * M, MPI_FLOAT,
			0, MPI_COMM_WORLD, &request_a);
	MPI_Ibcast(b, M * N, MPI_FLOAT, 0, MPI_COMM_WORLD, &request_b);

	MPI_Wait(&request_a, &status_a);
	MPI_Wait(&request_b, &status_b);

	omp_set_num_threads(NUM_THREADS);

	#pragma omp parallel for
	for(int i = 0; i < local_N; i++)
		for(int j = 0; j < N; j++)
			c[i * N + j] = 0.0f;

	#pragma omp parallel for
	for(int i = 0; i < local_N; i++)
		for(int k = 0; k < M; k++) {
			register float local_a = a[i * M + k];

			for (int j = 0; j < N; j++)
				c[i * N + j] += local_a * b[k * N + j];
		}

	MPI_Gather(c, local_N * N, MPI_FLOAT, c, local_N * N, MPI_FLOAT,
			0, MPI_COMM_WORLD);
}
