#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define GROUP_TILE_SIZE 64
#define ITEM_TILE_ROW_SIZE 4
#define ITEM_TILE_COL_SIZE 4
#define MAX_DEV 4

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

//====================================================================
// a[N][M] x b[M][N] = c[N][N]
//====================================================================

cl_int code;
cl_uint ndev;
cl_command_queue command_queue[MAX_DEV];
cl_mem bufferA[MAX_DEV], bufferB[MAX_DEV], bufferC[MAX_DEV];
cl_kernel kernel[MAX_DEV];
size_t sizeA[MAX_DEV], sizeB, sizeC[MAX_DEV];
int s[MAX_DEV], e[MAX_DEV];

void error(char *name, cl_int code) {
	printf("\nerror in %s, code: %d\n", name, code);
	exit(0);
}

void mat_mul(float *a, float *b, float *c, int N, int M) {

	/* a[K][M] x b[M][N] = c[K][N] */
	for(int i = 0; i < ndev; i++)
		if(s[i] != e[i]) {
			if((code = clEnqueueWriteBuffer(command_queue[i], bufferA[i], CL_FALSE, 0, sizeA[i], a + s[i] * M, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueWriteBufferA", code);
			if((code = clEnqueueWriteBuffer(command_queue[i], bufferB[i], CL_FALSE, 0, sizeB, b, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueWriteBufferB", code);
		}

	for(int i = 0; i < ndev; i++)
		if(s[i] != e[i]) {
			size_t global[2] = {N / ITEM_TILE_COL_SIZE, (e[i] - s[i]) / ITEM_TILE_ROW_SIZE};
			size_t local[2] = {GROUP_TILE_SIZE / ITEM_TILE_COL_SIZE, GROUP_TILE_SIZE / ITEM_TILE_ROW_SIZE};

			if((code = clEnqueueNDRangeKernel(command_queue[i], kernel[i], 2, NULL, global, local, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueNDRangeKernel", code);
		}

	for(int i = 0; i < ndev; i++)
		if(s[i] != e[i])
			if((code = clEnqueueReadBuffer(command_queue[i], bufferC[i], CL_TRUE, 0, sizeC[i], c + s[i] * N, 0, NULL, NULL)) != CL_SUCCESS)
				error("clEnqueueReadBufferC", code);
}

void setup(int N, int M) {
	/* do some runtime setup here */
	cl_platform_id platform;
	if((code = clGetPlatformIDs(1, &platform, NULL)) != CL_SUCCESS)
		error("clGetPlatformIDs", code);

	cl_device_id device[MAX_DEV];
	if((code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &ndev)) != CL_SUCCESS)
		error("clGetDeviceIDs", code);
	if((code = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, ndev, device, NULL)) != CL_SUCCESS)
		error("clGetDeviceIDs", code);

	cl_context context;
	if((context = clCreateContext(NULL, ndev, device, NULL, NULL, &code)) == NULL)
		error("clCreateContext", code);

	for(int i = 0; i < ndev; i++) {
		s[i] = N / GROUP_TILE_SIZE * i / ndev * GROUP_TILE_SIZE;
		e[i] = N / GROUP_TILE_SIZE * (i + 1) / ndev * GROUP_TILE_SIZE;
		sizeA[i] = (e[i] - s[i]) * M * sizeof(float);
		sizeC[i] = (e[i] - s[i]) * N * sizeof(float);
	
		if(s[i] != e[i])
			if((command_queue[i] = clCreateCommandQueue(context, device[i], 0, &code)) == NULL)
				error("clCreateCommandQueue", code);
	}

	sizeB = M * N * sizeof(float);
	for(int i = 0; i < ndev; i++)
		if(s[i] != e[i]) {
			if((bufferA[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeA[i], NULL, &code)) == NULL)
				error("clCreateBufferA", code);
			if((bufferB[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeB, NULL, &code)) == NULL)
				error("clCreateBufferB", code);
			if((bufferC[i] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC[i], NULL, &code)) == NULL)
				error("clCreateBufferC", code);
		}

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

	if((code = clBuildProgram(program, ndev, device, NULL, NULL, NULL)) != CL_SUCCESS) {
		if(code == CL_BUILD_PROGRAM_FAILURE)
			for(int i = 0; i < ndev; i++) {
				cl_int code2;
				size_t log_size;
				if((code2 = clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size)) != CL_SUCCESS)
					error("clGetProgramBuildInfo", code2);

				char *log = malloc(log_size);
				if((code2 = clGetProgramBuildInfo(program, device[i], CL_PROGRAM_BUILD_LOG, log_size, log, NULL)) != CL_SUCCESS)
					error("clGetProgramBuildInfo", code2);

				printf("\n%s\n", log);
			}

		error("clBuildProgram", code);
	}

	for(int i = 0; i < ndev; i++)
		if(s[i] != e[i])
			if((kernel[i] = clCreateKernel(program, "mat_mul", &code)) == NULL)
				error("clCreateKernel", code);

	for(int i = 0; i < ndev; i++)
		if(s[i] != e[i]) {
			if((code = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), (void *) &bufferA[i])) != CL_SUCCESS)
				error("clSetKernelArg0", code);
			if((code = clSetKernelArg(kernel[i], 1, sizeof(cl_mem), (void *) &bufferB[i])) != CL_SUCCESS)
				error("clSetKernelArg1", code);
			if((code = clSetKernelArg(kernel[i], 2, sizeof(cl_mem), (void *) &bufferC[i])) != CL_SUCCESS)
				error("clSetKernelArg2", code);
			if((code = clSetKernelArg(kernel[i], 3, sizeof(cl_int), (void *) &N)) != CL_SUCCESS)
				error("clSetKernelArg3", code);
			if((code = clSetKernelArg(kernel[i], 4, sizeof(cl_int), (void *) &M)) != CL_SUCCESS)
				error("clSetKernelArg4", code);
		}
}
