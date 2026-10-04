#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

// Функция ядра нахождения максимума
__global__ void findMax(const double* __restrict__ a, const double* __restrict__ b, double*  __restrict__ result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    while (idx < n) {
        result[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
        idx += totalThreads;
    }
}

// Хэлпер вывода ошибки от cuda
bool checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::cerr << "ERROR: CUDA error during " << operation
                  << ": " << cudaGetErrorString(error) << '\n';
        return false; 
    }
    return true;
}

// Функция освобождения памяти для cpu
template <typename... Pointers>
void closeCPU(Pointers*... pointers) {
    (delete[] pointers, ...);
}

// Функция освобождения памяти для gpu
template <typename... Pointers>
void closeGPU(Pointers*... pointers) {
    (cudaFree(pointers), ...);
}

int main() {
    // Настройки
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout << std::scientific << std::setprecision(4);
    
    // ============ НАЧАЛО ===============
    // Вводим данные
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

    // ============ НАСТРОЙКА CUDA ===============
    // Выделение памяти
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

    // Копирование памяти
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
    
    // Создание временных точек
    cudaEvent_t start, stop;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate start")) {
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }
    if (!checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop")) {
        cudaEventDestroy(start);
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    // ============ ЗАПУСК CUDA ===============
    std::cout << "Вычисления начались " << n << '\n'; 
    const int THREADS = 1024, BLOCKS = 1024;
    if (!checkCuda(cudaEventRecord(start), "cudaEventRecord start")) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }
    findMax<<<BLOCKS, THREADS>>>(d_a, d_b, d_result, n);

    // Проверяем, успешно ли запустилось ядро
    if (!checkCuda(cudaGetLastError(), "kernel launch")) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    // Временная стоп-точка
    if (!checkCuda(cudaEventRecord(stop), "cudaEventRecord stop")) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    // Ждём завершения ядра и проверяем ошибки во время выполнения
    if (!checkCuda(cudaEventSynchronize(stop), "kernel execution")) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    // ============ РЕЗУЛЬТАТЫ ===============
    // Копируем результат
    if (!checkCuda(cudaMemcpy(result, d_result, n*sizeof(double), cudaMemcpyDeviceToHost), "copy result to CPU"))  {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    // Время выполнения
    float milliseconds = 0;
    if (!checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapseTime not calculated")) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        closeGPU(d_a, d_b, d_result);
        closeCPU(a, b, result);
        return 0;
    }

    // Вывод
    // for (int i = 0; i < n; ++i) {
    //     std::cout << result[i] << " ";
    // }
    std::cout << '\n' << "Time [ms]: " << milliseconds << '\n';

    // Очистка
    cudaEventDestroy(start); cudaEventDestroy(stop);
    closeGPU(d_a, d_b, d_result);
    closeCPU(a, b, result);

    return 0;
}