#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <immintrin.h>
#define THREAD_MAX 16

//===========================================================================
// a[N][M] x b[M][N] = c[N][N]
//===========================================================================

// tile size: 32 * 32

static float *a, *b, *b_t, *c;
static int N, M, N_tile, M_tile;

static void *thread_transpose(void *thread_num_) {

	int thread_num = *((int *)thread_num_);
	int row_tile_start = M_tile * thread_num / THREAD_MAX;
	int row_tile_end = M_tile * (thread_num + 1) / THREAD_MAX;

	for(int row_tile = row_tile_start; row_tile < row_tile_end; row_tile++) {
		int row_start = row_tile * 32;
		int row_end = (row_tile + 1) * 32;
		
		float *start_addr_t = b + N * row_start;
		float *end_addr_t = b_t + row_start * 8;

		for(int col_tile = 0; col_tile < N_tile; col_tile++) {
			int col_start = col_tile * 32;
			float *start_addr = start_addr_t + col_start;
			float *end_addr = end_addr_t + M * col_start;
			
			for(int i = row_start; i < row_end; i++) {
				__m256 f1 = _mm256_loadu_ps(start_addr);
				__m256 f2 = _mm256_loadu_ps(start_addr + 8);
				__m256 f3 = _mm256_loadu_ps(start_addr + 16);
				__m256 f4 = _mm256_loadu_ps(start_addr + 24);
				start_addr += N;

				_mm256_storeu_ps(end_addr, f1);
				_mm256_storeu_ps(end_addr + M * 8, f2);
				_mm256_storeu_ps(end_addr + M * 16, f3);
				_mm256_storeu_ps(end_addr + M * 24, f4);
				end_addr += 8;
			}
		}
	}

	return NULL;
}

static void *thread_mat_mul(void *thread_num_) {

	int thread_num = *((int *)thread_num_);
	int row_tile_start = N_tile * thread_num / THREAD_MAX;
	int row_tile_end = N_tile * (thread_num + 1) / THREAD_MAX;

	for(int row_tile = row_tile_start; row_tile < row_tile_end; row_tile++) {
		int row_start = row_tile * 32;
		int row_end = (row_tile + 1) * 32;

		float *c_addr_t = c + N * row_start;

		for(int col_tile = 0; col_tile < N_tile; col_tile++) {
			int col_start = col_tile * 32;

			float *a_addr = a + M * row_start;
			float *c_addr = c_addr_t + col_start;

			for(int i = row_start; i < row_end; i++) {
				__m256 f6 = _mm256_setzero_ps();
				__m256 f7 = _mm256_setzero_ps();
				__m256 f8 = _mm256_setzero_ps();
				__m256 f9 = _mm256_setzero_ps();

				float *b_addr = b_t + M * col_start;

				for(int j = 0; j < M; j++) {
					__m256 f1 = _mm256_broadcast_ss(a_addr);
					a_addr++;
					__m256 f2 = _mm256_loadu_ps(b_addr);
					__m256 f3 = _mm256_loadu_ps(b_addr + M * 8);
					__m256 f4 = _mm256_loadu_ps(b_addr + M * 16);
					__m256 f5 = _mm256_loadu_ps(b_addr + M * 24);
					b_addr += 8;

					f2 = _mm256_mul_ps(f1, f2);
					f3 = _mm256_mul_ps(f1, f3);
					f4 = _mm256_mul_ps(f1, f4);
					f5 = _mm256_mul_ps(f1, f5);

					f6 = _mm256_add_ps(f6, f2);
					f7 = _mm256_add_ps(f7, f3);
					f8 = _mm256_add_ps(f8, f4);
					f9 = _mm256_add_ps(f9, f5);
				}

				_mm256_storeu_ps(c_addr, f6);
				_mm256_storeu_ps(c_addr + 8, f7);
				_mm256_storeu_ps(c_addr + 16, f8);
				_mm256_storeu_ps(c_addr + 24, f9);
				c_addr += N;
			}
		}
	}

	return NULL;
}

void mat_mul(float *a_, float *b_, float *c_, int N_, int M_) {	

	pthread_t thread[THREAD_MAX];
	int thread_num[THREAD_MAX];

	N = N_;
	M = M_;
	a = a_;
	b = b_;
	c = c_;
	N_tile = N / 32;
	M_tile = M / 32;

	// transpose
	b_t = (float *)malloc(N * M * sizeof(float));
	for(int i = 0; i < THREAD_MAX; i++) {
		thread_num[i] = i;
		if(pthread_create(&thread[i], NULL, thread_transpose, (void *)(&thread_num[i])))
			printf("error\n");
	}

	// join
	for(int i = 0; i < THREAD_MAX; i++)
		if(pthread_join(thread[i], NULL))
			printf("eror\n");

	// multiplication
	for(int i = 0; i < THREAD_MAX; i++) {
		if(pthread_create(&thread[i], NULL, thread_mat_mul, (void *)(&thread_num[i])))
			printf("error\n");
	}

	// join
	for(int i = 0; i < THREAD_MAX; i++)
		if(pthread_join(thread[i], NULL))
			printf("error\n");

	free(b_t);
}
