CC = mpicc
TARGET = mat_mul
OBJECTS = main.o mul.o timer.o

CFLAGS = -std=c11 -Wall -O3 -fopenmp
LDFLAGS = -lm -fopenmp

MODE = mpi
DEVICE = cpu
NODES = 4
SLOTS = 1

NDIM = 4096
MDIM = 4096
EXE_OPTION =

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean run

clean:
	rm -rf $(TARGET) $(OBJECTS) task*

run: $(TARGET)
	@thorq --add --mode $(MODE) --nodes $(NODES) --slots $(SLOTS) --device $(DEVICE) \
	 ./$^ $(NDIM) $(MDIM) $(EXE_OPTION)

ref: mat_mul_ref
	@thorq --add --mode $(MODE) --nodes $(NODES) --slots $(SLOTS) --device $(DEVICE) \
	 ./$^ $(NDIM) $(MDIM) $(EXE_OPTION)

opt: mat_mul_opt
	@thorq --add --mode $(MODE) --nodes $(NODES) --slots $(SLOTS) --device $(DEVICE) \
	 ./$^ $(NDIM) $(MDIM) $(EXE_OPTION)
