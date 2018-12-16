#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define NUM_THREADS 32

#include "colorizer.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <sys/time.h>
#include <unistd.h>

#include <CL/cl.h>
#include <errno.h>
#include <omp.h>

cl_int code;
cl_command_queue command_queue;
cl_mem buffer_in, buffer_out, buffer_weight, buffer_bias;
cl_kernel kernel_112x112_1, kernel_112x112_2, kernel_56x56_1, kernel_56x56_2;
cl_kernel kernel_28x28_1, kernel_28x28_2, kernel_14x14_1, kernel_14x14_2, kernel_7x7_1;

void error(char *name, cl_int code) {
	printf("\nerror in %s, code: %d\n", name, code);
	exit(0);
}

/*
double get_time() {
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec + tv.tv_usec * 1e-6;
}
*/

static void conv(float *in, float *out, float *weight, float *bias, int H, int W, int K, int C, int stride) {

	if (H == 224 && stride == 2) {
		int HOUT = H / 2, WOUT = W / 2;
		#pragma omp parallel for
		for (int k = 0; k < K; ++k)
			for (int hout = 0; hout < HOUT; ++hout)
				for (int wout = 0; wout < WOUT; ++wout) {
					float sum = bias[k];
					for (int c = 0; c < C; ++c)
						for (int r = 0; r < 3; ++r)
							for (int s = 0; s < 3; ++s) {
								int h = hout * 2 + r - 1;
								int w = wout * 2 + s - 1;
								if (0 <= h && h < H && 0 <= w && w < W)
									sum += in[c * H * W + h * W + w] * weight[k * C * 3 * 3 + c * 3 * 3 + r * 3 + s];
							}
					out[k * HOUT * WOUT + hout * WOUT + wout] = sum;
				}
	}
	else {
		if((code = clEnqueueWriteBuffer(command_queue, buffer_in, CL_FALSE, 0, C * H * W * sizeof(float), in, 0, NULL, NULL)) != CL_SUCCESS)
			error("clEnqueueWriteBuffer_in", code);
		if((code = clEnqueueWriteBuffer(command_queue, buffer_weight, CL_FALSE, 0, K * C * 3 * 3 * sizeof(float), weight, 0, NULL, NULL)) != CL_SUCCESS)
			error("clEnqueueWriteBuffer_weight", code);
		if((code = clEnqueueWriteBuffer(command_queue, buffer_bias, CL_FALSE, 0, K * sizeof(float), bias, 0, NULL, NULL)) != CL_SUCCESS)
			error("clEnqueueWriteBuffer_bias", code);

		if (H == 112 && stride == 1) {
			if((code = clSetKernelArg(kernel_112x112_1, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 14, 14}, local[2] = {14, 14};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_112x112_1, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 112 * 112 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);	
		}
		else if (H == 112 && stride == 2) {
			if((code = clSetKernelArg(kernel_112x112_2, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 14, 14}, local[2] = {14, 14};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_112x112_2, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 56 * 56 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);	
		}
		else if (H == 56 && stride == 1) {
			if((code = clSetKernelArg(kernel_56x56_1, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 14, 14}, local[2] = {14, 14};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_56x56_1, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 56 * 56 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);
		}
		else if (H == 56 && stride == 2) {
			if((code = clSetKernelArg(kernel_56x56_2, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 14, 14}, local[2] = {14, 14};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_56x56_2, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 28 * 28 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);	
		}
		else if (H == 28 && stride == 1) {
			if((code = clSetKernelArg(kernel_28x28_1, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 14, 14}, local[2] = {14, 14};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_28x28_1, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 28 * 28 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);
		
		}
		else if (H == 28 && stride == 2) {
			if((code = clSetKernelArg(kernel_28x28_2, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 14, 14}, local[2] = {14, 14};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_28x28_2, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 14 * 14 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);
		
		}
		else if (H == 14 && stride == 1) {
			if((code = clSetKernelArg(kernel_14x14_1, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 14, 14}, local[2] = {14, 14};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_14x14_1, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 14 * 14 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);
		
		}
		else if (H == 14 && stride == 2) {
			if((code = clSetKernelArg(kernel_14x14_2, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 7, 7}, local[2] = {7, 7};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_14x14_2, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 7 * 7 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);
		
		}
		else if (H == 7 && stride == 1) {
			if((code = clSetKernelArg(kernel_7x7_1, 4, sizeof(cl_int), (void *) &C)) != CL_SUCCESS)
				error("clSetKernelArg4", code);

			size_t global[2] = {K * 7, 7}, local[2] = {7, 7};
			if((code = clEnqueueNDRangeKernel(command_queue, kernel_7x7_1, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);

			if((code = clEnqueueReadBuffer(command_queue, buffer_out, CL_TRUE, 0, K * 7 * 7 * sizeof(float), out, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBuffer_out", code);
		}
	}
}

static void fc(float *in, float *out, float *weight, float *bias, int K, int C) {
	#pragma omp parallel for
	for (int k = 0; k < K; ++k) {
		float s = 0;
		for (int c = 0; c < C; ++c) {
			s += in[c] * weight[k * C + c];
		}
		s += bias[k];
		out[k] = s;
	}
}

static void relu(float *inout, int CHW) {
	#pragma omp parallel for
	for (int chw = 0; chw < CHW; ++chw) {
		inout[chw] = fmaxf(inout[chw], 0);
	}
}

static void sigmoid(float *inout, int CHW) {
	#pragma omp parallel for
	for (int chw = 0; chw < CHW; ++chw) {
		inout[chw] = 1 / (1 + expf(-inout[chw]));
	}
}

static void fuse(float *ml, float *gf, float *out) {
	#pragma omp parallel for
	for (int k = 0; k < 256; ++k) {
		for (int h = 0; h < 28; ++h) {
			for (int w = 0; w < 28; ++w) {
				out[k * 28 * 28 + h * 28 + w] = ml[k * 28 * 28 + h * 28 + w];
			}
		}
	}
	#pragma omp parallel for
	for (int k = 256; k < 512; ++k) {
		for (int h = 0; h < 28; ++h) {
			for (int w = 0; w < 28; ++w) {
				out[k * 28 * 28 + h * 28 + w] = gf[k - 256];
			}
		}
	}
}

static void upsample(float *in, float *out, int H, int W, int C) {
	int HOUT = 2 * H, WOUT = 2 * W;
	#pragma omp parallel for
	for (int c = 0; c < C; ++c) {
		for (int h = 0; h < H; ++h) {
			for (int w = 0; w < W; ++w) {
				float t = in[c * H * W + h * W + w];
				out[c * HOUT * WOUT + (2 * h + 0) * WOUT + (2 * w + 0)] = t;
				out[c * HOUT * WOUT + (2 * h + 0) * WOUT + (2 * w + 1)] = t;
				out[c * HOUT * WOUT + (2 * h + 1) * WOUT + (2 * w + 0)] = t;
				out[c * HOUT * WOUT + (2 * h + 1) * WOUT + (2 * w + 1)] = t;
			}
		}
	}
}

void colorizer_init() {

	omp_set_num_threads(NUM_THREADS);

	cl_platform_id platform;
	if((code = clGetPlatformIDs(1, &platform, NULL)) != CL_SUCCESS)
		error("clGetPlatformIDs", code);

	cl_device_id device;
	if((code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL)) != CL_SUCCESS)
		error("clGetDeviceIDs", code);

	cl_context context;
	if((context = clCreateContext(NULL, 1, &device, NULL, NULL, &code)) == NULL)
		error("clCreateContext", code);

	if((command_queue = clCreateCommandQueue(context, device, 0, &code)) == NULL)
		error("clCreateCommandQueue", code);

	if((buffer_in = clCreateBuffer(context, CL_MEM_READ_ONLY, 1605632 * sizeof(float), NULL, &code)) == NULL)
		error("clCreateBuffer_in", code);
	if((buffer_out = clCreateBuffer(context, CL_MEM_READ_ONLY, 1605632 * sizeof(float), NULL, &code)) == NULL)
		error("clCreateBuffer_out", code);
	if((buffer_weight = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 2359296 * sizeof(float), NULL, &code)) == NULL)
		error("clCreateBuffer_weight", code);
	if((buffer_bias = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 512 * sizeof(float), NULL, &code)) == NULL)
		error("clCreateBuffer_bias", code);


	FILE *fp;
	if((fp = fopen("kernel.cl", "r")) == NULL)
		error("fopen", errno);

	if(fseek(fp, 0, SEEK_END))
		error("fseek", errno);

	long fpos;
	if((fpos = ftell(fp)) == -1)
		error("ftell", errno);
	
	rewind(fp);

	size_t kernel_src_len = fpos;
	char *kernel_src = malloc(kernel_src_len * sizeof(char));
	fread(kernel_src, kernel_src_len, 1, fp);

	cl_program program;
	if((program = clCreateProgramWithSource(context, 1, (const char **) &kernel_src, &kernel_src_len, &code)) == NULL)
		error("clCreateProgramWithSource", code);

	if((code = clBuildProgram(program, 1, &device, NULL, NULL, NULL)) != CL_SUCCESS) {
		if(code == CL_BUILD_PROGRAM_FAILURE) {
			cl_int code2;
			size_t log_size;
			if((code2 = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size)) != CL_SUCCESS)
				error("clGetProgramBuildInfo", code2);

			char *log = malloc(log_size);
			if((code2 = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL)) != CL_SUCCESS)
				error("clGetProgramBuildInfo", code2);

			printf("\n%s\n", log);
		}

		error("clBuildProgram", code);
	}

	if((kernel_112x112_1 = clCreateKernel(program, "conv_112x112_1", &code)) == NULL)
		error("clCreateKernel_112x112_1", code);
	if((kernel_112x112_2 = clCreateKernel(program, "conv_112x112_2", &code)) == NULL)
		error("clCreateKernel_112x112_2", code);	
	if((kernel_56x56_1 = clCreateKernel(program, "conv_56x56_1", &code)) == NULL)
		error("clCreateKernel_56x56_1", code);
	if((kernel_56x56_2 = clCreateKernel(program, "conv_56x56_2", &code)) == NULL)
		error("clCreateKernel_56x56_2", code);
	if((kernel_28x28_1 = clCreateKernel(program, "conv_28x28_1", &code)) == NULL)
		error("clCreateKernel_28x28_1", code);
	if((kernel_28x28_2 = clCreateKernel(program, "conv_28x28_2", &code)) == NULL)
		error("clCreateKernel_28x28_2", code);
	if((kernel_14x14_1 = clCreateKernel(program, "conv_14x14_1", &code)) == NULL)
		error("clCreateKernel_14x14_1", code);
	if((kernel_14x14_2 = clCreateKernel(program, "conv_14x14_2", &code)) == NULL)
		error("clCreateKernel_14x14_2", code);
	if((kernel_7x7_1 = clCreateKernel(program, "conv_7x7_1", &code)) == NULL)
		error("clCreateKernel_7x7_1", code);

	if((code = clSetKernelArg(kernel_112x112_1, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_112x112_1, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_112x112_1, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_112x112_1, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);

	if((code = clSetKernelArg(kernel_112x112_2, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_112x112_2, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_112x112_2, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_112x112_2, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);

	if((code = clSetKernelArg(kernel_56x56_1, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_56x56_1, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_56x56_1, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_56x56_1, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);


	if((code = clSetKernelArg(kernel_56x56_2, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_56x56_2, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_56x56_2, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_56x56_2, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);


	if((code = clSetKernelArg(kernel_28x28_1, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_28x28_1, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_28x28_1, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_28x28_1, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);


	if((code = clSetKernelArg(kernel_28x28_2, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_28x28_2, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_28x28_2, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_28x28_2, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);


	if((code = clSetKernelArg(kernel_14x14_1, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_14x14_1, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_14x14_1, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_14x14_1, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);


	if((code = clSetKernelArg(kernel_14x14_2, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_14x14_2, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_14x14_2, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_14x14_2, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);


	if((code = clSetKernelArg(kernel_7x7_1, 0, sizeof(cl_mem), (void *) &buffer_in)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel_7x7_1, 1, sizeof(cl_mem), (void *) &buffer_out)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel_7x7_1, 2, sizeof(cl_mem), (void *) &buffer_weight)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel_7x7_1, 3, sizeof(cl_mem), (void *) &buffer_bias)) != CL_SUCCESS)
		error("clSetKernelArg3", code);
}

void colorizer(int nimg, float *network, float *inputs, float *outputs) {

	float *ll_conv1_w = network; network += 64 * 1 * 3 * 3;
	float *ll_conv1_b = network; network += 64;
	float *ll_conv2_w = network; network += 128 * 64 * 3 * 3;
	float *ll_conv2_b = network; network += 128;
	float *ll_conv3_w = network; network += 128 * 128 * 3 * 3;
	float *ll_conv3_b = network; network += 128;
	float *ll_conv4_w = network; network += 256 * 128 * 3 * 3;
	float *ll_conv4_b = network; network += 256;
	float *ll_conv5_w = network; network += 256 * 256 * 3 * 3;
	float *ll_conv5_b = network; network += 256;
	float *ll_conv6_w = network; network += 512 * 256 * 3 * 3;
	float *ll_conv6_b = network; network += 512;
	float *ml_conv1_w = network; network += 512 * 512 * 3 * 3;
	float *ml_conv1_b = network; network += 512;
	float *ml_conv2_w = network; network += 256 * 512 * 3 * 3;
	float *ml_conv2_b = network; network += 256;
	float *gf_conv1_w = network; network += 512 * 512 * 3 * 3;
	float *gf_conv1_b = network; network += 512;
	float *gf_conv2_w = network; network += 512 * 512 * 3 * 3;
	float *gf_conv2_b = network; network += 512;
	float *gf_conv3_w = network; network += 512 * 512 * 3 * 3;
	float *gf_conv3_b = network; network += 512;
	float *gf_conv4_w = network; network += 512 * 512 * 3 * 3;
	float *gf_conv4_b = network; network += 512;
	float *gf_fc1_w = network; network += 1024 * 25088;
	float *gf_fc1_b = network; network += 1024;
	float *gf_fc2_w = network; network += 512 * 1024;
	float *gf_fc2_b = network; network += 512;
	float *gf_fc3_w = network; network += 256 * 512;
	float *gf_fc3_b = network; network += 256;
	float *co_conv1_w = network; network += 256 * 512 * 3 * 3;
	float *co_conv1_b = network; network += 256;
	float *co_conv2_w = network; network += 128 * 256 * 3 * 3;
	float *co_conv2_b = network; network += 128;
	float *co_conv3_w = network; network += 64 * 128 * 3 * 3;
	float *co_conv3_b = network; network += 64;
	float *co_conv4_w = network; network += 64 * 64 * 3 * 3;
	float *co_conv4_b = network; network += 64;
	float *co_conv5_w = network; network += 32 * 64 * 3 * 3;
	float *co_conv5_b = network; network += 32;
	float *co_conv6_w = network; network += 2 * 32 * 3 * 3;
	float *co_conv6_b = network; network += 2;

	float *ll_fm1 = (float*)malloc(64 * 112 * 112 * sizeof(float));
	float *ll_fm2 = (float*)malloc(128 * 112 * 112 * sizeof(float));
	float *ll_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
	float *ll_fm4 = (float*)malloc(256 * 56 * 56 * sizeof(float));
	float *ll_fm5 = (float*)malloc(256 * 28 * 28 * sizeof(float));
	float *ll_fm6 = (float*)malloc(512 * 28 * 28 * sizeof(float));
	float *ml_fm1 = (float*)malloc(512 * 28 * 28 * sizeof(float));
	float *ml_fm2 = (float*)malloc(256 * 28 * 28 * sizeof(float));
	float *gf_fm1 = (float*)malloc(512 * 14 * 14 * sizeof(float));
	float *gf_fm2 = (float*)malloc(512 * 14 * 14 * sizeof(float));
	float *gf_fm3 = (float*)malloc(512 * 7 * 7 * sizeof(float));
	float *gf_fm4 = (float*)malloc(512 * 7 * 7 * sizeof(float));
	float *gf_fm5 = (float*)malloc(1024 * sizeof(float));
	float *gf_fm6 = (float*)malloc(512 * sizeof(float));
	float *gf_fm7 = (float*)malloc(256 * sizeof(float));
	float *ml_gf_fused_fm = (float*)malloc(512 * 28 * 28 * sizeof(float));
	float *co_fm1 = (float*)malloc(256 * 28 * 28 * sizeof(float));
	float *co_fm2 = (float*)malloc(128 * 28 * 28 * sizeof(float));
	float *co_fm3 = (float*)malloc(128 * 56 * 56 * sizeof(float));
	float *co_fm4 = (float*)malloc(64 * 56 * 56 * sizeof(float));
	float *co_fm5 = (float*)malloc(64 * 56 * 56 * sizeof(float));
	float *co_fm6 = (float*)malloc(64 * 112 * 112 * sizeof(float));
	float *co_fm7 = (float*)malloc(32 * 112 * 112 * sizeof(float));

	for (int n = 0; n < nimg; ++n) {
		float *input = inputs + n * 224 * 224;
		float *output = outputs + n * 2 * 112 * 112;

//		double start = get_time();
		conv(input, ll_fm1, ll_conv1_w, ll_conv1_b, 224, 224, 64, 1, 2);
//		double end = get_time();
//		printf("conv input -> ll_fm1\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(ll_fm1, 64 * 112 * 112);
//		end = get_time();
//		printf("relu ll_fm1\n%lf ms\n", 1000 * (end - start));
		
//		start = get_time();
		conv(ll_fm1, ll_fm2, ll_conv2_w, ll_conv2_b, 112, 112, 128, 64, 1);
//		end = get_time();
//		printf("conv ll_fm1 -> ll_fm2\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(ll_fm2, 128 * 112 * 112);
//		end = get_time();
//		printf("relu ll_fm2\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(ll_fm2, ll_fm3, ll_conv3_w, ll_conv3_b, 112, 112, 128, 128, 2);
//		end = get_time();
//		printf("conv ll_fm2 -> ll_fm3\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(ll_fm3, 128 * 56 * 56);
//		end = get_time();
//		printf("relu ll_fm3\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(ll_fm3, ll_fm4, ll_conv4_w, ll_conv4_b, 56, 56, 256, 128, 1);
//		end = get_time();
//		printf("conv ll_fm3 -> ll_fm4\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(ll_fm4, 256 * 56 * 56);
//		end = get_time();
//		printf("relu ll_fm4\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(ll_fm4, ll_fm5, ll_conv5_w, ll_conv5_b, 56, 56, 256, 256, 2);
//		end = get_time();
//		printf("conv ll_fm4 -> ll_fm5\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(ll_fm5, 256 * 28 * 28);
//		end = get_time();
//		printf("relu ll_fm5\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(ll_fm5, ll_fm6, ll_conv6_w, ll_conv6_b, 28, 28, 512, 256, 1);
//		end = get_time();
//		printf("conv ll_fm5 -> ll_fm6\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(ll_fm6, 512 * 28 * 28);
//		end = get_time();
//		printf("relu ll_fm6\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(ll_fm6, ml_fm1, ml_conv1_w, ml_conv1_b, 28, 28, 512, 512, 1);
//		end = get_time();
//		printf("conv ll_fm6 -> ml_fm1\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(ml_fm1, 512 * 28 * 28);
//		end = get_time();
//		printf("relu ml_fm1\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(ml_fm1, ml_fm2, ml_conv2_w, ml_conv2_b, 28, 28, 256, 512, 1);
//		end = get_time();
//		printf("conv ml_fm1 -> ml_fm2\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(ml_fm2, 256 * 28 * 28);
//		end = get_time();
//		printf("relu ml_fm2\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(ll_fm6, gf_fm1, gf_conv1_w, gf_conv1_b, 28, 28, 512, 512, 2);
//		end = get_time();
//		printf("conv ll_fm6 -> gf_fm1\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(gf_fm1, 512 * 14 * 14);
//		end = get_time();
//		printf("relu gf_fm1\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(gf_fm1, gf_fm2, gf_conv2_w, gf_conv2_b, 14, 14, 512, 512, 1);
//		end = get_time();
//		printf("conv gf_fm1 -> gf_fm2\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(gf_fm2, 512 * 14 * 14);
//		end = get_time();
//		printf("relu gf_fm2\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(gf_fm2, gf_fm3, gf_conv3_w, gf_conv3_b, 14, 14, 512, 512, 2);
//		end = get_time();
//		printf("conv gf_fm2 -> gf_fm3\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(gf_fm3, 512 * 7 * 7);
//		end = get_time();
//		printf("relu gf_fm3\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(gf_fm3, gf_fm4, gf_conv4_w, gf_conv4_b, 7, 7, 512, 512, 1);
//		end = get_time();
//		printf("conv gf_fm3 -> gf_fm4\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(gf_fm4, 512 * 7 * 7);
//		end = get_time();
//		printf("relu gf_fm4\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		fc(gf_fm4, gf_fm5, gf_fc1_w, gf_fc1_b, 1024, 25088);
//		end = get_time();
//		printf("fc gf_fm4 -> gf_fm5\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(gf_fm5, 1024);
//		end = get_time();
//		printf("relu ll_fm5\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		fc(gf_fm5, gf_fm6, gf_fc2_w, gf_fc2_b, 512, 1024);
//		end = get_time();
//		printf("fc gf_fm5 -> gf_fm6\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(gf_fm6, 512);
//		end = get_time();
//		printf("relu gf_fm6\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		fc(gf_fm6, gf_fm7, gf_fc3_w, gf_fc3_b, 256, 512);
//		end = get_time();
//		printf("fc gf_fm6 -> gf_fm7\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(gf_fm7, 256);
//		end = get_time();
//		printf("relu gf_fm7\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		fuse(ml_fm2, gf_fm7, ml_gf_fused_fm);
//		end = get_time();
//		printf("fuse ml_fm2 & gf_fm7 -> ml_gf_fused_fm\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(ml_gf_fused_fm, co_fm1, co_conv1_w, co_conv1_b, 28, 28, 256, 512, 1);
//		end = get_time();
//		printf("conv ml_gf_fused_fm -> co_fm1\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(co_fm1, 256 * 28 * 28);
//		end = get_time();
//		printf("relu co_fm1\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(co_fm1, co_fm2, co_conv2_w, co_conv2_b, 28, 28, 128, 256, 1);
//		end = get_time();
//		printf("conv co_fm1 -> co_fm2\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(co_fm2, 128 * 28 * 28);
//		end = get_time();
//		printf("relu co_fm2\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		upsample(co_fm2, co_fm3, 28, 28, 128);
//		end = get_time();
//		printf("upsample co_fm2 -> co_fm3\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(co_fm3, co_fm4, co_conv3_w, co_conv3_b, 56, 56, 64, 128, 1);
//		end = get_time();
//		printf("conv co_fm3 -> co_fm4\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(co_fm4, 64 * 56 * 56);
//		end = get_time();
//		printf("relu co_fm4\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(co_fm4, co_fm5, co_conv4_w, co_conv4_b, 56, 56, 64, 64, 1);
//		end = get_time();
//		printf("conv co_fm4 -> co_fm5\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(co_fm5, 64 * 56 * 56);
//		end = get_time();
//		printf("relu co_fm5\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		upsample(co_fm5, co_fm6, 56, 56, 64);
//		end = get_time();
//		printf("upsample co_fm5 -> co_fm6\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(co_fm6, co_fm7, co_conv5_w, co_conv5_b, 112, 112, 32, 64, 1);
//		end = get_time();
//		printf("conv co_fm6 -> co_fm7\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		relu(co_fm7, 32 * 112 * 112);
//		end = get_time();
//		printf("relu co_fm7\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		conv(co_fm7, output, co_conv6_w, co_conv6_b, 112, 112, 2, 32, 1);
//		end = get_time();
//		printf("conv co_fm7 -> output\n%lf ms\n", 1000 * (end - start));

//		start = get_time();
		sigmoid(output, 2 * 112 * 112);
//		end = get_time();
//		printf("sigmoid output\n%lf ms\n", 1000 * (end - start));
	}
}
