CC = gcc
TARGET = mat_mul
OBJECTS = main.o mul.o timer.o

CFLAGS = -std=c11 -Wall -O3
LDFLAGS = -lm -lOpenCL

MODE = single
DEVICE = gpu/1080

NDIM = 8192
MDIM = 8192
EXE_OPTION =

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean run

clean:
	rm -rf $(TARGET) $(OBJECTS) task*

run: $(TARGET)
	@thorq --add --mode $(MODE) --device $(DEVICE) --timeout 60 ./$^ $(NDIM) $(MDIM) $(EXE_OPTION)

ref: mat_mul_ref_1080
	@thorq --add --mode $(MODE) --device $(DEVICE) --timeout 60 ./$^ $(NDIM) $(MDIM) $(EXE_OPTION)

opt: mat_mul_opt_1080
	@thorq --add --mode $(MODE) --device $(DEVICE) --timeout 60 ./$^ $(NDIM) $(MDIM) $(EXE_OPTION)
