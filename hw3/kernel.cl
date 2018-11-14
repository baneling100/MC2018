#define GROUP_TILE_SIZE 64
#define ITEM_TILE_ROW_SIZE 4
#define ITEM_TILE_COL_SIZE 8
#define ROW_GAP (GROUP_TILE_SIZE / ITEM_TILE_ROW_SIZE)
#define COL_GAP (GROUP_TILE_SIZE / ITEM_TILE_COL_SIZE)

__kernel void mat_mul(const __global float *a, const __global float *b, __global float *c, const int N, const int M) {

	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int global_col = get_group_id(0) * GROUP_TILE_SIZE + local_col;
	const int global_row = get_group_id(1) * GROUP_TILE_SIZE + local_row;

	__local float local_a[GROUP_TILE_SIZE][GROUP_TILE_SIZE];
	__local float local_b[GROUP_TILE_SIZE][GROUP_TILE_SIZE];

	float acc[ITEM_TILE_ROW_SIZE][ITEM_TILE_COL_SIZE] = {};

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

		barrier(CLK_LOCAL_MEM_FENCE);

		for(int j = 0; j < GROUP_TILE_SIZE; j++)
			for(int k = 0; k < ITEM_TILE_ROW_SIZE; k++)
				for(int l = 0; l < ITEM_TILE_COL_SIZE; l++)
					acc[k][l] += local_a[j][ROW_GAP * k + local_row] * local_b[j][COL_GAP * l + local_col];

		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	for(int i = 0; i < ITEM_TILE_ROW_SIZE; i++) {
		const int global_tile_row = ROW_GAP * i + global_row;

		for(int j = 0; j < ITEM_TILE_COL_SIZE; j++) {
			const int global_tile_col = COL_GAP * j + global_col;

			c[global_tile_row * N + global_tile_col] = acc[i][j];
		}
	}
}
