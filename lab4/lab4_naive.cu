#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <iomanip>
#include <string>
#include <exception>
#include <cmath>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/extrema.h>

constexpr double EPS = 1e-7;

constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 16;
constexpr int GRID_X = 32;
constexpr int GRID_Y = 32;
constexpr int SWAP_BLOCK = 256;
constexpr int SWAP_GRID = 32;

// ============================================================
// ХЭЛПЕРЫ
// ============================================================

// Хэлпер вывода ошибки
void formatError(const std::string& context, const std::string& err) {
    std::cerr << "ERROR: " << context << ": " << err << '\n';
}

// Хэлпер вывода ошибки от CUDA
bool checkCuda(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        formatError("CUDA error during " + operation,
                    cudaGetErrorString(error));
        return false;
    }
    return true;
}

// ============================================================
// ЯДРА CUDA
// ============================================================

// Преобразователь для поиска максимального элемента
// в столбце обычной матрицы через Thrust.
struct ColumnAbs {
    const double* A;
    int n;
    int k;

    __host__ __device__
    double operator()(int offset) const {
        return fabs(A[(size_t)(k + offset) * n + k]);
    }
};

// Перестановка строк обычной матрицы
__global__ void swapRows(double* A, int n, int r1, int r2) {
    if (r1 == r2)
        return;

    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < n;
         j += blockDim.x * gridDim.x) {

        double tmp = A[(size_t)r1 * n + j];

        A[(size_t)r1 * n + j] =
            A[(size_t)r2 * n + j];

        A[(size_t)r2 * n + j] = tmp;
    }
}

// Обновление обычной матрицы
__global__ void naive(double* A, int n, int k) {
    int i = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int j = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;

    int strideY = blockDim.y * gridDim.y;
    int strideX = blockDim.x * gridDim.x;

    for (; i < n; i += strideY) {
        for (int col = j; col < n; col += strideX) {
            double factor = A[(size_t)i * n + k]
                          / A[(size_t)k * n + k];

            A[(size_t)i * n + col] -=
                factor * A[(size_t)k * n + col];
        }
    }
}

// ============================================================
// ИНЫЕ ФУНКЦИИ
// ============================================================

// Поиск главного элемента через THRUST
int findPivotNaive(const double* d_A, int n, int k) {
    auto countingBegin = thrust::make_counting_iterator(0);

    ColumnAbs absColumn{d_A, n, k};

    auto begin = thrust::make_transform_iterator(
        countingBegin, absColumn
    );

    auto end = begin + (n - k);

    auto maxIt = thrust::max_element(
        thrust::device, begin, end
    );

    return k + (int)(maxIt - begin);
}

// Вычисление определителя
double determinantDirect(
    const std::vector<double>& diagonal,
    int swaps
) {
    double result = (swaps % 2 == 0) ? 1.0 : -1.0;

    for (double x : diagonal) {
        result *= x;
    }

    return result;
}

// ============================================================
// CPU
// ============================================================

int main() {
   // ============ НАЧАЛО ===============
    int n = 0;
    std::cin >> n;
    if (!std::cin || n <= 0 || n > 10000) {
        formatError("Input", "n is forbidden");
        return 0;
    }

    size_t matrixSize = (size_t)n * n;
    std::vector<double> matrix(matrixSize);
    for (size_t i = 0; i < matrixSize; ++i) {
        std::cin >> matrix[i];
        if (!std::cin) {
            formatError("Input", "not enough matrix elements");
            return 0;
        }
    }

    // ============ НАСТРОЙКА CUDA ===============
    double* d_matrix = nullptr;
    if (!checkCuda(
        cudaMalloc((void**)&d_matrix, matrixSize * sizeof(double)),
        "cudaMalloc d_matrix"
    )) {
        return 0;
    }

    if (!checkCuda(
        cudaMemcpy(d_matrix, matrix.data(), matrixSize * sizeof(double), cudaMemcpyHostToDevice),
        "copy matrix to GPU"
    )) {
        cudaFree(d_matrix);
        return 0;
    }

    // ============ МЕТОД ГАУССА ===============
    int swaps = 0;
    bool singular = false;
    bool cudaOk = true;

    for (int k = 0; k < n; ++k) {
        // Ищем главный элемент в текущем столбце.
        int pivotRow = -1;
        try {
            pivotRow = findPivotNaive(d_matrix, n, k);
        }
        catch (const std::exception& e) {
            formatError("Thrust pivot search", e.what());
            cudaOk = false;
            break;
        }

        double maxAbs = 0.0;
        if (!checkCuda(
            cudaMemcpy(&maxAbs, d_matrix + (size_t)pivotRow * n + k, sizeof(double), cudaMemcpyDeviceToHost
            ),
            "copy pivot to CPU"
        )) {
            cudaOk = false;
            break;
        }
        maxAbs = std::abs(maxAbs);
        if (maxAbs < EPS) {
            singular = true;
            break;
        }
        
        if (k == n-1) continue;

        // Переставляем строки исходной матрицы.
        if (pivotRow != k) {
            swapRows<<<SWAP_GRID, SWAP_BLOCK>>>(d_matrix, n, k, pivotRow);
            if (!checkCuda(cudaGetLastError(), "swap kernel launch")) {
                cudaOk = false;
                break;
            }
            ++swaps;
        }
       
        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid(GRID_X, GRID_Y);
        naive<<<grid, block>>>(d_matrix, n, k);

        if (!checkCuda(
            cudaGetLastError(),
            "elimination kernel launch"
        )) {
            cudaOk = false;
            break;
        }
    }

    // Ждём завершения всех ядер.
    if (cudaOk) {
        if (!checkCuda(cudaDeviceSynchronize(), "kernel execution")) {
            cudaOk = false;
            cudaFree(d_matrix);
            return 0;
        }
    }

    // ============ РЕЗУЛЬТАТЫ ===============
    std::vector<double> diagonal(n, 0.0);
    if (cudaOk && !singular) {
        for (int i = 0; i < n; ++i) {
            if (!checkCuda(cudaMemcpy(&diagonal[i], d_matrix + (size_t)i * n + i, sizeof(double), cudaMemcpyDeviceToHost),
                "copy diagonal element to CPU"
            )) {
                cudaFree(d_matrix);
                return 0;
            }
        }
    }

    double determinant = 0.0;
    if (cudaOk && !singular) {
        determinant = determinantDirect(diagonal, swaps);
    }
    std::cout << std::scientific << std::setprecision(10) << determinant << '\n';

    if (!cudaOk) {
        formatError("Result", "computation was not completed");
    }

    cudaFree(d_matrix);

    return 0;
}
