#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void findMax(const double* __restrict__ a, const double* __restrict__ b, double*  __restrict__ result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    while (idx < n) {
        result[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
        idx += totalThreads;
    }
}

bool checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "ERROR: CUDA error during " << operation
                  << ": " << cudaGetErrorString(error) << '\n';
        return false; 
    }
    return true;
}

template <typename... Pointers>
void closeCPU(Pointers*... pointers) {
    (delete[] pointers, ...);
}

template <typename... Pointers>
void closeGPU(Pointers*... pointers) {
    (cudaFree(pointers), ...);
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout << std::scientific << std::setprecision(10);
    
    int n;
    std::cin >> n;
    if (n <= 0 || n >= (1 << 25)) {
        std::cerr << "ERROR: n out of range\n";
        return 0;
    }

    double* a = new double[n];
    double* b = new double[n];
    double* result = new double[n];
    for (int i = 0; i < n; ++i) {
        std::cin >> a[i];
    }
    for (int i = 0; i < n; ++i) {
        std::cin >> b[i];
    }

    double* d_a; double* d_b; double* d_result;
    if (!checkCuda(cudaMalloc(&d_a, n*sizeof(double)), "cudaMalloc d_a")) {
        closeCPU(a, b, result);
        return 0;
    }
    if (!checkCuda(cudaMalloc(&d_b, n*sizeof(double)), "cudaMalloc d_b")) {
        closeGPU(d_a);
        closeCPU(a, b, result);
        return 0;
    }
    if (!checkCuda(cudaMalloc(&d_result, n*sizeof(double)), "cudaMalloc d_result")) {
        closeGPU(d_a, d_b);
        closeCPU(a, b, result);
        return 0;
    }
    if (!checkCuda(cudaMemcpy(d_a, a, n*sizeof(double), cudaMemcpyHostToDevice), "copy a to GPU")) {
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }
    if (!checkCuda(cudaMemcpy(d_b, b, n*sizeof(double), cudaMemcpyHostToDevice), "copy b to GPU")) {
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    const int THREADS = 256, BLOCKS = 1024;
    findMax<<<BLOCKS, THREADS>>>(d_a, d_b, d_result, n);

    // Проверяем, успешно ли запустилось ядро
    if (!checkCuda(cudaGetLastError(), "kernel launch")) {
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    // Ждём завершения ядра и проверяем ошибки во время выполнения
    if (!checkCuda(cudaDeviceSynchronize(), "kernel execution")) {
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    if (!checkCuda(cudaMemcpy(result, d_result, n*sizeof(double), cudaMemcpyDeviceToHost), "copy result to CPU"))  {
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    for (int i = 0; i < n; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << '\n';

    closeGPU(d_a, d_b, d_result);
    closeCPU(a, b, result);

    return 0;
}