CC = gcc
TARGET = mat_mul
OBJECTS = main.o mul.o timer.o

CFLAGS = -std=c11 -Wall -O3
LDFLAGS = -lm -lOpenCL

MODE = single
DEVICE = gpu/7970

NDIM = 16384
MDIM = 16384
EXE_OPTION =

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

.PHONY: clean run

clean:
	rm -rf $(TARGET) $(OBJECTS) task*

run: $(TARGET)
	@thorq --add --mode $(MODE) --device $(DEVICE) --timeout 60 ./$(TARGET) $(NDIM) $(MDIM) $(EXE_OPTION)
