// matrix_mul_dynamic.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel CUDA per il prodotto tra matrici quadrate double N×N
__global__ void matrixMul(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Uso: %s <dimensione matrice quadrata>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "Errore: la dimensione deve essere un intero positivo\n");
        return EXIT_FAILURE;
    }

    size_t size = N * N * sizeof(double);

    // Alloca memoria sull'host
    double *h_A = (double*)malloc(size);
    double *h_B = (double*)malloc(size);
    double *h_C = (double*)malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Errore allocazione host memory\n");
        return EXIT_FAILURE;
    }

    // Inizializza matrici
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0;
        h_B[i] = 2.0;
    }

    // Alloca memoria sul device
    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Calcola griglia e blocchi
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + 15) / 16, (N + 15) / 16);

    matrixMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Controlla il primo elemento per verifica
    printf("C[0][0] = %f\n", h_C[0]);

    // Libera memoria
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
