#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <exception>

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
// Преобразователь для поиска максимального по модулю
// элемента в строке транспонированной матрицы через Thrust.
struct ColumnAbs {
    const double* T;
    int n;
    int k;

    __host__ __device__
    double operator()(int offset) const {
        return fabs(T[(size_t)k * n + (k + offset)]);
    }
};

// Перестановка столбцов транспонированной матрицы
__global__ void swapTransposed(double* T, int n, int c1, int c2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; j < n; j += stride) {
        if (c1 != c2) {
            double tmp = T[(size_t)j * n + c1];

            T[(size_t)j * n + c1] =
                T[(size_t)j * n + c2];

            T[(size_t)j * n + c2] = tmp;
        }
    }
}


// Обновление транспонированной матрицы
__global__ void halfRefinement(double* T, int n, int k) {
    // Координаты потока
    int j = k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;

    // Шаг перемещения по сетке
    int strideY = blockDim.y * gridDim.y;
    int strideX = blockDim.x * gridDim.x;

    // Проходим по строкам транспонированной матрицы
    for (; j < n; j += strideY) {
        double factor = T[(size_t)j * n + k] / T[(size_t)k * n + k];
        // Проходим по столбцам транспонированной матрицы
        for (int col = i; col < n; col += strideX) {
            T[(size_t)j * n + col] -= factor * T[(size_t)k * n + col];
        }
    }
}

// ============================================================
// ИНЫЕ ФУНКЦИИ
// ============================================================
// Поиск главного элемента через THRUST
int findPivotTransposed(const double* d_T, int n, int k) {
    auto countingBegin = thrust::make_counting_iterator(0);
    ColumnAbs absColumn{d_T, n, k};

    auto begin = thrust::make_transform_iterator(countingBegin, absColumn);
    auto end = begin + (n - k);
    auto maxIt = thrust::max_element(thrust::device, begin, end);

    return k + (int)(maxIt - begin);
}

// Вычисление определителя
double determinantLog(const std::vector<double>& diagonal, int swaps) {
    double logSum = 0.0;
    int sign = ((swaps & 1) == 0) ? 1 : -1;
    for (double x : diagonal) {
        if (x < 0.0)
            sign = -sign;

        logSum += std::log(std::abs(x));
    }

    return (double)sign * std::exp(logSum);
}

// ============================================================
// CPU
// ============================================================

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
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

    std::vector<double> hostMatrix(matrixSize);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            hostMatrix[(size_t)j * n + i] = matrix[(size_t)i * n + j];
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
        cudaMemcpy(d_matrix, hostMatrix.data(), matrixSize * sizeof(double), cudaMemcpyHostToDevice),
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
            pivotRow = findPivotTransposed(d_matrix, n, k);
        }
        catch (const std::exception& e) {
            formatError("Thrust pivot search", e.what());
            cudaOk = false;
            break;
        }

        double maxAbs = 0.0;
        if (!checkCuda(
            cudaMemcpy(&maxAbs, d_matrix + (size_t)k * n + pivotRow, sizeof(double), cudaMemcpyDeviceToHost),
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
            swapTransposed<<<SWAP_GRID, SWAP_BLOCK>>>(d_matrix, n, k, pivotRow);
            if (!checkCuda(cudaGetLastError(), "swap kernel launch")) {
                cudaOk = false;
                break;
            }
            ++swaps;
        }

        dim3 block(BLOCK_X, BLOCK_Y);
        dim3 grid(GRID_X, GRID_Y);
        halfRefinement<<<grid, block>>>(d_matrix, n, k);

        if (!checkCuda(cudaGetLastError(), "elimination kernel launch")) {
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
        determinant = determinantLog(diagonal, swaps);
    }
    std::cout << std::scientific << std::setprecision(10) << determinant << '\n';

    if (!cudaOk) {
        formatError("Result", "computation was not completed");
    }

    cudaFree(d_matrix);

    return 0;
}
