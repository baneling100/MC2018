CC = gcc
EXECUTABLE = project
CCFILES = main.c timer.c colorizer.c
OBJECTS = main.o timer.o colorizer.o

CFLAGS = -std=gnu99 -Ofast -Wall -mavx -fopenmp
LDFLAGS = -lm -lrt -lOpenCL -fopenmp

INPUTFILE = input2.bin
OUTPUTFILE = output2.bin

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $^ $(LDFLAGS)

run: $(EXECUTABLE)
	@thorq --add --mode single --device cpu ./$(EXECUTABLE) $(INPUTFILE) $(OUTPUTFILE)

clean:
	rm -rf $(EXECUTABLE) $(OBJECTS) $(OUTPUTFILE) task*
