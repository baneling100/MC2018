__kernel void conv_112x112_1(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 14 x 14
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[58][114];
	__local float weight[3][3];
	__local float bias;
	float sum[8][8] = {0.0f,};

	in[4 * local_row][0] = 0.0f;
	in[4 * local_row + 1][0] = 0.0f;
	in[4 * local_row + 2][0] = 0.0f;
	in[4 * local_row + 3][0] = 0.0f;
	in[4 * local_row + 4][0] = 0.0f;
	in[4 * local_row + 5][0] = 0.0f;
	in[4 * local_row][113] = 0.0f;
	in[4 * local_row + 1][113] = 0.0f;
	in[4 * local_row + 2][113] = 0.0f;
	in[4 * local_row + 3][113] = 0.0f;
	in[4 * local_row + 4][113] = 0.0f;
	in[4 * local_row + 5][113] = 0.0f;

	for (int c = 0; c < C; c++) {
		in[0][8 * local_col + 1] = 0.0f;
		in[0][8 * local_col + 2] = 0.0f;
		in[0][8 * local_col + 3] = 0.0f;
		in[0][8 * local_col + 4] = 0.0f;
		in[0][8 * local_col + 5] = 0.0f;
		in[0][8 * local_col + 6] = 0.0f;
		in[0][8 * local_col + 7] = 0.0f;
		in[0][8 * local_col + 8] = 0.0f;

		for (int i = 0; i < 4; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 8; j++) {
				const int x = j * 14 + local_col;
				in[y + 1][x + 1] = in_[c * 112 * 112 + y * 112 + x];
			}
		}

		in[57][8 * local_col + 1] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col];
		in[57][8 * local_col + 2] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 1];
		in[57][8 * local_col + 3] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 2];
		in[57][8 * local_col + 4] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 3];
		in[57][8 * local_col + 5] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 4];
		in[57][8 * local_col + 6] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 5];
		in[57][8 * local_col + 7] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 6];
		in[57][8 * local_col + 8] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 7];

		const int p = local_row / 5, q = local_col / 5;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
		
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 0; i < 4; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 8; j++) {
				const int x = j * 14 + local_col;
				for (int r = 0; r < 3; r++)
					for (int s = 0; s < 3; s++)
						sum[i][j] += in[y + r][x + s] * weight[r][s];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		in[0][8 * local_col + 1] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col];
		in[0][8 * local_col + 2] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 1];
		in[0][8 * local_col + 3] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 2];
		in[0][8 * local_col + 4] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 3];
		in[0][8 * local_col + 5] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 4];
		in[0][8 * local_col + 6] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 5];
		in[0][8 * local_col + 7] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 6];
		in[0][8 * local_col + 8] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 7];

		for (int i = 4; i < 8; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 8; j++) {
				const int x = j * 14 + local_col;
				in[y - 56 + 1][x + 1] = in_[c * 112 * 112 + y * 112 + x];
			}
		}

		in[57][8 * local_col + 1] = 0.0f;
		in[57][8 * local_col + 2] = 0.0f;
		in[57][8 * local_col + 3] = 0.0f;
		in[57][8 * local_col + 4] = 0.0f;
		in[57][8 * local_col + 5] = 0.0f;
		in[57][8 * local_col + 6] = 0.0f;
		in[57][8 * local_col + 7] = 0.0f;
		in[57][8 * local_col + 8] = 0.0f;
	
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 4; i < 8; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 8; j++) {
				const int x = j * 14 + local_col;
				for (int r = 0; r < 3; r++)
					for (int s = 0; s < 3; s++)
						sum[i][j] += in[y - 56 + r][x + s] * weight[r][s];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	for (int i = 0; i < 8; i++) {
		const int y = i * 14 + local_row;
		for (int j = 0; j < 8; j++) {
			const int x = j * 14 + local_col;
			out[k * 112 * 112 + y * 112 + x] = sum[i][j] + bias;
		}
	}
}

__kernel void conv_112x112_2(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 14 x 14
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[58][114];
	__local float weight[3][3];
	__local float bias;
	float sum[4][4] = {0.0f,};

	in[4 * local_row][0] = 0.0f;
	in[4 * local_row + 1][0] = 0.0f;
	in[4 * local_row + 2][0] = 0.0f;
	in[4 * local_row + 3][0] = 0.0f;
	in[4 * local_row + 4][0] = 0.0f;
	in[4 * local_row + 5][0] = 0.0f;
	in[4 * local_row][113] = 0.0f;
	in[4 * local_row + 1][113] = 0.0f;
	in[4 * local_row + 2][113] = 0.0f;
	in[4 * local_row + 3][113] = 0.0f;
	in[4 * local_row + 4][113] = 0.0f;
	in[4 * local_row + 5][113] = 0.0f;

	for (int c = 0; c < C; c++) {
		in[0][8 * local_col + 1] = 0.0f;
		in[0][8 * local_col + 2] = 0.0f;
		in[0][8 * local_col + 3] = 0.0f;
		in[0][8 * local_col + 4] = 0.0f;
		in[0][8 * local_col + 5] = 0.0f;
		in[0][8 * local_col + 6] = 0.0f;
		in[0][8 * local_col + 7] = 0.0f;
		in[0][8 * local_col + 8] = 0.0f;

		for (int i = 0; i < 4; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 8; j++) {
				const int x = j * 14 + local_col;
				in[y + 1][x + 1] = in_[c * 112 * 112 + y * 112 + x];
			}
		}

		in[57][8 * local_col + 1] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col];
		in[57][8 * local_col + 2] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 1];
		in[57][8 * local_col + 3] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 2];
		in[57][8 * local_col + 4] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 3];
		in[57][8 * local_col + 5] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 4];
		in[57][8 * local_col + 6] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 5];
		in[57][8 * local_col + 7] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 6];
		in[57][8 * local_col + 8] = in_[c * 112 * 112 + 56 * 112 + 8 * local_col + 7];
	
		const int p = local_row / 5, q = local_col / 5;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
		
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 0; i < 2; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 4; j++) {
				const int x = j * 14 + local_col;
				for (int r = 0; r < 3; r++)
					for (int s = 0; s < 3; s++)
						sum[i][j] += in[y * 2 + r][x * 2 + s] * weight[r][s];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		in[0][8 * local_col + 1] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col];
		in[0][8 * local_col + 2] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 1];
		in[0][8 * local_col + 3] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 2];
		in[0][8 * local_col + 4] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 3];
		in[0][8 * local_col + 5] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 4];
		in[0][8 * local_col + 6] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 5];
		in[0][8 * local_col + 7] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 6];
		in[0][8 * local_col + 8] = in_[c * 112 * 112 + 55 * 112 + 8 * local_col + 7];

		for (int i = 4; i < 8; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 8; j++) {
				const int x = j * 14 + local_col;
				in[y - 56 + 1][x + 1] = in_[c * 112 * 112 + y * 112 + x];
			}
		}

		in[57][8 * local_col + 1] = 0.0f;
		in[57][8 * local_col + 2] = 0.0f;
		in[57][8 * local_col + 3] = 0.0f;
		in[57][8 * local_col + 4] = 0.0f;
		in[57][8 * local_col + 5] = 0.0f;
		in[57][8 * local_col + 6] = 0.0f;
		in[57][8 * local_col + 7] = 0.0f;
		in[57][8 * local_col + 8] = 0.0f;
	
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 2; i < 4; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 4; j++) {
				const int x = j * 14 + local_col;
				for (int r = 0; r < 3; r++)
					for (int s = 0; s < 3; s++)
						sum[i][j] += in[y * 2 - 56 + r][x * 2 + s] * weight[r][s];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	for (int i = 0; i < 4; i++) {
		const int y = i * 14 + local_row;
		for (int j = 0; j < 4; j++) {
			const int x = j * 14 + local_col;
			out[k * 56 * 56 + y * 56 + x] = sum[i][j] + bias;
		}
	}
}

__kernel void conv_56x56_1(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 14 x 14
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[58][58];
	__local float weight[3][3];
	__local float bias;
	float sum[4][4] = {0.0f,};

	in[0][4 * local_col] = 0.0f;
	in[0][4 * local_col + 1] = 0.0f;
	in[0][4 * local_col + 2] = 0.0f;
	in[0][4 * local_col + 3] = 0.0f;
	in[0][4 * local_col + 4] = 0.0f;
	in[57][4 * local_col + 1] = 0.0f;
	in[57][4 * local_col + 2] = 0.0f;
	in[57][4 * local_col + 3] = 0.0f;
	in[57][4 * local_col + 4] = 0.0f;
	in[57][4 * local_col + 5] = 0.0f;
	in[4 * local_row + 1][0] = 0.0f;
	in[4 * local_row + 2][0] = 0.0f;
	in[4 * local_row + 3][0] = 0.0f;
	in[4 * local_row + 4][0] = 0.0f;
	in[4 * local_row + 5][0] = 0.0f;
	in[4 * local_row][57] = 0.0f;
	in[4 * local_row + 1][57] = 0.0f;
	in[4 * local_row + 2][57] = 0.0f;
	in[4 * local_row + 3][57] = 0.0f;
	in[4 * local_row + 4][57] = 0.0f;

	for (int c = 0; c < C; c++) {
		for (int i = 0; i < 4; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 4; j++) {
				const int x = j * 14 + local_col;
				in[y + 1][x + 1] = in_[c * 56 * 56 + y * 56 + x];
			}
		}

		const int p = local_row / 5, q = local_col / 5;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 0; i < 4; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 4; j++) {
				const int x = j * 14 + local_col;
				for (int r = 0; r < 3; r++)
					for (int s = 0; s < 3; s++)
						sum[i][j] += in[y + r][x + s] * weight[r][s];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	for (int i = 0; i < 4; i++) {
		const int y = i * 14 + local_row;
		for (int j = 0; j < 4; j++) {
			const int x = j * 14 + local_col;
			out[k * 56 * 56 + y * 56 + x] = sum[i][j] + bias;
		}
	}
}

__kernel void conv_56x56_2(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 14 x 14
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[58][58];
	__local float weight[3][3];
	__local float bias;
	float sum[2][2] = {0.0f,};

	in[0][4 * local_col] = 0.0f;
	in[0][4 * local_col + 1] = 0.0f;
	in[0][4 * local_col + 2] = 0.0f;
	in[0][4 * local_col + 3] = 0.0f;
	in[0][4 * local_col + 4] = 0.0f;
	in[57][4 * local_col + 1] = 0.0f;
	in[57][4 * local_col + 2] = 0.0f;
	in[57][4 * local_col + 3] = 0.0f;
	in[57][4 * local_col + 4] = 0.0f;
	in[57][4 * local_col + 5] = 0.0f;
	in[4 * local_row + 1][0] = 0.0f;
	in[4 * local_row + 2][0] = 0.0f;
	in[4 * local_row + 3][0] = 0.0f;
	in[4 * local_row + 4][0] = 0.0f;
	in[4 * local_row + 5][0] = 0.0f;
	in[4 * local_row][57] = 0.0f;
	in[4 * local_row + 1][57] = 0.0f;
	in[4 * local_row + 2][57] = 0.0f;
	in[4 * local_row + 3][57] = 0.0f;
	in[4 * local_row + 4][57] = 0.0f;

	for (int c = 0; c < C; c++) {
		for (int i = 0; i < 4; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 4; j++) {
				const int x = j * 14 + local_col;
				in[y + 1][x + 1] = in_[c * 56 * 56 + y * 56 + x];
			}
		}

		const int p = local_row / 5, q = local_col / 5;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 0; i < 2; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 2; j++) {
				const int x = j * 14 + local_col;
				for (int r = 0; r < 3; r++)
					for (int s = 0; s < 3; s++)
						sum[i][j] += in[y * 2 + r][x * 2 + s] * weight[r][s];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	for (int i = 0; i < 2; i++) {
		const int y = i * 14 + local_row;
		for (int j = 0; j < 2; j++) {
			const int x = j * 14 + local_col;
			out[k * 28 * 28 + y * 28 + x] = sum[i][j] + bias;
		}
	}
}

__kernel void conv_28x28_1(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 14 x 14
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[30][30];
	__local float weight[3][3];
	__local float bias;
	float sum[2][2] = {0.0f,};

	in[0][2 * local_col] = 0.0f;
	in[0][2 * local_col + 1] = 0.0f;
	in[0][2 * local_col + 2] = 0.0f;
	in[29][2 * local_col + 1] = 0.0f;
	in[29][2 * local_col + 2] = 0.0f;
	in[29][2 * local_col + 3] = 0.0f;
	in[2 * local_row + 1][0] = 0.0f;
	in[2 * local_row + 2][0] = 0.0f;
	in[2 * local_row + 3][0] = 0.0f;
	in[2 * local_row][29] = 0.0f;
	in[2 * local_row + 1][29] = 0.0f;
	in[2 * local_row + 2][29] = 0.0f;

	for (int c = 0; c < C; c++) {
		for (int i = 0; i < 2; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 2; j++) {
				const int x = j * 14 + local_col;
				in[y + 1][x + 1] = in_[c * 28 * 28 + y * 28 + x];
			}
		}

		const int p = local_row / 5, q = local_col / 5;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 0; i < 2; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 2; j++) {
				const int x = j * 14 + local_col;
				for (int r = 0; r < 3; r++)
					for (int s = 0; s < 3; s++)
						sum[i][j] += in[y + r][x + s] * weight[r][s];
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	for (int i = 0; i < 2; i++) {
		const int y = i * 14 + local_row;
		for (int j = 0; j < 2; j++) {
			const int x = j * 14 + local_col;
			out[k * 28 * 28 + y * 28 + x] = sum[i][j] + bias;
		}
	}
}

__kernel void conv_28x28_2(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 14 x 14
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[30][30];
	__local float weight[3][3];
	__local float bias;
	float sum = 0.0f;

	in[0][2 * local_col] = 0.0f;
	in[0][2 * local_col + 1] = 0.0f;
	in[0][2 * local_col + 2] = 0.0f;
	in[29][2 * local_col + 1] = 0.0f;
	in[29][2 * local_col + 2] = 0.0f;
	in[29][2 * local_col + 3] = 0.0f;
	in[2 * local_row + 1][0] = 0.0f;
	in[2 * local_row + 2][0] = 0.0f;
	in[2 * local_row + 3][0] = 0.0f;
	in[2 * local_row][29] = 0.0f;
	in[2 * local_row + 1][29] = 0.0f;
	in[2 * local_row + 2][29] = 0.0f;

	for (int c = 0; c < C; c++) {
		for (int i = 0; i < 2; i++) {
			const int y = i * 14 + local_row;
			for (int j = 0; j < 2; j++) {
				const int x = j * 14 + local_col;
				in[y + 1][x + 1] = in_[c * 28 * 28 + y * 28 + x];
			}
		}

		const int p = local_row / 5, q = local_col / 5;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int r = 0; r < 3; r++)
			for (int s = 0; s < 3; s++)
				sum += in[2 * local_row + r][2 * local_col + s] * weight[r][s];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	out[k * 14 * 14 + local_row * 14 + local_col] = sum + bias;
}

__kernel void conv_14x14_1(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 14 x 14
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[16][16];
	__local float weight[3][3];
	__local float bias;
	float sum = 0.0f;

	in[0][local_col] = 0.0f;
	in[0][local_col + 1] = 0.0f;
	in[0][local_col + 2] = 0.0f;
	in[15][local_col + 1] = 0.0f;
	in[15][local_col + 2] = 0.0f;
	in[15][local_col + 3] = 0.0f;
	in[local_row + 1][0] = 0.0f;
	in[local_row + 2][0] = 0.0f;
	in[local_row + 3][0] = 0.0f;
	in[local_row][15] = 0.0f;
	in[local_row + 1][15] = 0.0f;
	in[local_row + 2][15] = 0.0f;

	for (int c = 0; c < C; c++) {
		in[local_row + 1][local_col + 1] = in_[c * 14 * 14 + local_row * 14 + local_col];
		
		const int p = local_row / 5, q = local_col / 5;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int r = 0; r < 3; r++)
			for (int s = 0; s < 3; s++)
				sum += in[local_row + r][local_col + s] * weight[r][s];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	out[k * 14 * 14 + local_row * 14 + local_col] = sum + bias;
}

__kernel void conv_14x14_2(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 7 x 7
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[16][16];
	__local float weight[3][3];
	__local float bias;
	float sum = 0.0f;

	in[0][2 * local_col] = 0.0f;
	in[0][2 * local_col + 1] = 0.0f;
	in[0][2 * local_col + 2] = 0.0f;
	in[15][2 * local_col + 1] = 0.0f;
	in[15][2 * local_col + 2] = 0.0f;
	in[15][2 * local_col + 3] = 0.0f;
	in[2 * local_row + 1][0] = 0.0f;
	in[2 * local_row + 2][0] = 0.0f;
	in[2 * local_row + 3][0] = 0.0f;
	in[2 * local_row][15] = 0.0f;
	in[2 * local_row + 1][15] = 0.0f;
	in[2 * local_row + 2][15] = 0.0f;

	for (int c = 0; c < C; c++) {
		for (int i = 0; i < 2; i++) {
			const int y = i * 7 + local_row;
			for (int j = 0; j < 2; j++) {
				const int x = j * 7 + local_col;
				in[y + 1][x + 1] = in_[c * 14 * 14 + y * 14 + x];
			}
		}

		const int p = local_row / 3, q = local_col / 3;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int r = 0; r < 3; r++)
			for (int s = 0; s < 3; s++)
				sum += in[2 * local_row + r][2 * local_col + s] * weight[r][s];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	out[k * 7 * 7 + local_row * 7 + local_col] = sum + bias;
}

__kernel void conv_7x7_1(const __global float *in_, __global float *out, const __global float *weight_, const __global float *bias_, const int C) { // 7 x 7
	const int local_col = get_local_id(0);
	const int local_row = get_local_id(1);
	const int k = get_group_id(0);
	__local float in[9][9];
	__local float weight[3][3];
	__local float bias;
	float sum = 0.0f;

	in[0][local_col] = 0.0f;
	in[0][local_col + 1] = 0.0f;
	in[0][local_col + 2] = 0.0f;
	in[8][local_col + 1] = 0.0f;
	in[8][local_col + 2] = 0.0f;
	in[8][local_col + 3] = 0.0f;
	in[local_row + 1][0] = 0.0f;
	in[local_row + 2][0] = 0.0f;
	in[local_row + 3][0] = 0.0f;
	in[local_row][8] = 0.0f;
	in[local_row + 1][8] = 0.0f;
	in[local_row + 2][8] = 0.0f;

	for (int c = 0; c < C; c++) {
		in[local_row + 1][local_col + 1] = in_[c * 7 * 7 + local_row * 7 + local_col];

		const int p = local_row / 3, q = local_col / 3;
		weight[p][q] = weight_[k * C * 3 * 3 + c * 3 * 3 + p * 3 + q];
	
		barrier(CLK_LOCAL_MEM_FENCE);

		for (int r = 0; r < 3; r++)
			for (int s = 0; s < 3; s++)
				sum += in[local_row + r][local_col + s] * weight[r][s];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	bias = bias_[k];

	out[k * 7 * 7 + local_row * 7 + local_col] = sum + bias;
}
