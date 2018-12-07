CC_SIMD = gcc
CC_CUDA = nvcc

SIMD = simd
CUDA = cuda

NDIM = 8192
MDIM = 8192
EXE_OPTION =

all: $(SIMD) $(CUDA)

$(SIMD): mat_mul_simd.c
	$(CC_SIMD) -o $@ $^ -std=c99 -Wall -O3 -mavx -ftree-vectorize -ffast-math -funroll-loops -lm

$(CUDA): mat_mul_cuda.cu
	$(CC_CUDA) -o $@ $^ -Wno-deprecated-gpu-targets -O3 -lm

.PHONY: clean runsimd runcuda

clean:
	rm -rf *.o $(SIMD) $(CUDA) task*

runsimd: $(SIMD)
	@thorq --add --mode single --device cpu --timeout 60 ./$< $(NDIM) $(MDIM) $(EXE_OPTION)

runcuda: $(CUDA)
	@thorq --add --mode single --device gpu/1080 --timeout 60 ./$< $(NDIM) $(MDIM) $(EXE_OPTION)

simdref: simd_ref
	@thorq --add --mode single --device cpu --timeout 60 ./$< $(NDIM) $(MDIM) $(EXE_OPTION)

cudaref: cuda_ref
	@thorq --add --mode single --device gpu/1080 --timeout 60 ./$< $(NDIM) $(MDIM) $(EXE_OPTION)
