#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_SAFECALL(call)                                             \
{                                                                       \
    call;                                                               \
    cudaError err = cudaGetLastError();                                 \
    if (cudaSuccess != err) {                                           \
        fprintf(                                                        \
            stderr,                                                     \
            "Cuda error in function '%s' file '%s' in line %i : %s.\n", \
            #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
        fflush(stderr);                                                 \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
}


__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) c[id] = a[id] + b[id];
}

int main(int argc, char *argv[])
{
    int n = 100000;
    if (argc > 1) n = atoi(argv[1]);

    double *h_a;
    double *h_b;
    double *h_c;

    double *d_a;
    double *d_b;
    double *d_c;

    size_t bytes = n * sizeof(double);

    h_a = (double *)malloc(bytes);
    h_b = h_a;
    h_c = h_a;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    for (int i = 0; i < n; i++)
        h_a[i] = sin(i) * sin(i);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < n; i++)
        h_b[i] = cos(i) * cos(i);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < n; i++)
        h_c[i] = 0;
    cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice);   

    int blockSize = 1024;
    int gridSize = (int)ceil((float)n / blockSize);

    CUDA_SAFECALL((vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n)));

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < n; i++) sum += h_c[i];
    printf("Final sum = %f; sum/n = %f (should be ~1)\n", sum, sum / n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);

    return 0;
}
