CUDA_LIB := /usr/local/cuda/lib64/
CUDA_LIBS := -lcudart
CC	= mpicc
CFLAGS	= -O0 -g -Wall -fopenmp
LDFLAGS = -g -lm -lpthread -lX11 -fopenmp -L$(CUDA_LIB) $(CUDA_LIBS)
VERBOSE	=
TARGET	= nbody_brute_force nbody_barnes_hut
OBJS	= ui.o xstuff.o nbody_tools.o nbody_alloc.o

DISPLAY = -DDISPLAY
DISPLAY =

DUMP = -DDUMP_RESULT
#DUMP =

all: $(TARGET)

nbody_brute_force_cuda: nbody_brute_force_cuda.cu
	nvcc -Xcompiler -fopenmp -o nbody_brute_force_cuda.o -c nbody_brute_force_cuda.cu

nbody_brute_force: nbody_brute_force.o $(OBJS) nbody_brute_force_cuda
	$(CC) $(VERBOSE) -o $@ $< $(OBJS) nbody_brute_force_cuda.o $(LDFLAGS)

nbody_barnes_hut: nbody_barnes_hut.o $(OBJS)
	$(CC) $(VERBOSE) -o $@ $< $(OBJS)  $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< $(VERBOSE) $(DISPLAY) $(DUMP)
clean:
	rm -f *.o *.log $(TARGET) 
