CC = mpicc
EXECUTABLE = project
CCFILES = main.c timer.c colorizer.c
OBJECTS = main.o timer.o colorizer.o

CFLAGS = -std=gnu99 -Ofast -Wall -mavx -fopenmp
LDFLAGS = -lm -lrt -lOpenCL -fopenmp

INPUTFILE = input3.bin
OUTPUTFILE = output3.bin

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

run: $(EXECUTABLE)
	@thorq --add --mode mpi --nodes 4 --device gpu/7970 ./$(EXECUTABLE) $(INPUTFILE) $(OUTPUTFILE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) $(OUTPUTFILE) task*
