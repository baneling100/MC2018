#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define GROUP_TILE_SIZE 64
#define ITEM_TILE_ROW_SIZE 4
#define ITEM_TILE_COL_SIZE 8

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

//====================================================================
// a[N][M] x b[M][N] = c[N][N]
//====================================================================

cl_int code;
cl_command_queue command_queue;
cl_mem bufferA, bufferB, bufferC;
cl_kernel kernel;

void error(char *name, cl_int code) {
	printf("\nerror in %s, code: %d\n", name, code);
	exit(0);
}

void mat_mul(float *a, float *b, float *c, int N, int M) {

	if((code = clEnqueueWriteBuffer(command_queue, bufferA, CL_FALSE, 0, N * M * sizeof(float), a, 0, NULL, NULL)) != CL_SUCCESS)
		error("clEnqueueWriteBufferA", code);
	if((code = clEnqueueWriteBuffer(command_queue, bufferB, CL_FALSE, 0, M * N * sizeof(float), b, 0, NULL, NULL)) != CL_SUCCESS)
		error("clEnqueueWriteBufferB", code);

	size_t global[2] = {N / ITEM_TILE_COL_SIZE, N / ITEM_TILE_ROW_SIZE}, local[2] = {GROUP_TILE_SIZE / ITEM_TILE_COL_SIZE, GROUP_TILE_SIZE / ITEM_TILE_ROW_SIZE};
	if((code = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
		error("clEnqueueNDRangeKernel", code);

	if((code = clEnqueueReadBuffer(command_queue, bufferC, CL_TRUE, 0, N * N * sizeof(float), c, 0, NULL, NULL)) != CL_SUCCESS)
		error("clEnqueueReadBufferC", code);
}

void setup(int N, int M) {
	/* do some runtime setup here */
	cl_platform_id platform;
	if((code = clGetPlatformIDs(1, &platform, NULL)) != CL_SUCCESS)
		error("clGetPlatformIDs", code);

	cl_device_id device;
	if((code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL)) != CL_SUCCESS)
		error("clGetDeviceIDs", code);

	cl_context context;
	if((context = clCreateContext(NULL, 1, &device, NULL, NULL, &code)) == NULL)
		error("clCreateContext", code);

	if((command_queue = clCreateCommandQueue(context, device, 0, &code)) == NULL)
		error("clCreateCommandQueue", code);

	if((bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, N * M * sizeof(float), NULL, &code)) == NULL)
		error("clCreateBufferA", code);
	if((bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, M * N * sizeof(float), NULL, &code)) == NULL)
		error("clCreateBufferB", code);
	if((bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * N * sizeof(float), NULL, &code)) == NULL)
		error("clCreateBufferC", code);

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

	if((kernel = clCreateKernel(program, "mat_mul", &code)) == NULL)
		error("clCreateKernel", code);

	if((code = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &bufferA)) != CL_SUCCESS)
		error("clSetKernelArg0", code);
	if((code = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &bufferB)) != CL_SUCCESS)
		error("clSetKernelArg1", code);
	if((code = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &bufferC)) != CL_SUCCESS)
		error("clSetKernelArg2", code);
	if((code = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *) &N)) != CL_SUCCESS)
		error("clSetKernelArg3", code);
	if((code = clSetKernelArg(kernel, 4, sizeof(cl_int), (void *) &M)) != CL_SUCCESS)
		error("clSetKernelArg4", code);
}
