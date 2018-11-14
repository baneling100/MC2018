CC = gcc
TARGET = mat_mul
OBJECTS = main.o mul.o timer.o

CFLAGS = -std=c99 -Wall -O3 -mavx -ftree-vectorize -ffast-math -funroll-loops
LDFLAGS = -lm -lpthread

MODE = single
DEVICE = cpu

NDIM = 2048
MDIM = 2048

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean run

clean:
	rm -rf $(TARGET) $(OBJECTS)

run: $(TARGET)
	@thorq --add --mode $(MODE) --device $(DEVICE) ./$(TARGET) $(NDIM) $(MDIM) -v
