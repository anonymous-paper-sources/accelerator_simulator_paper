ARCH=sm_70

all:
	nvcc -ccbin=$(CXX) -O3 -arch=$(ARCH) matmul.cu -o matmul

clean:
	rm -f matmul
