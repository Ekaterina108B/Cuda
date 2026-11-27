#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <exception>
#include <chrono>

constexpr double EPS = 1e-7;

// ============================================================
// ХЭЛПЕРЫ
// ============================================================

void formatError(const std::string& context, const std::string& err) {
    std::cerr << "ERROR: " << context << ": " << err << '\n';
}

// Поиск главного элемента по модулю в столбце k
// транспонированной матрицы T (размер n x n).
// В транспонированной матрице столбец k — это T[k*n + j], j >= k.
int findPivotTransposed(const std::vector<double>& T, int n, int k) {
    int best = k;
    double bestAbs = std::abs(T[(size_t)k * n + k]);
    for (int j = k + 1; j < n; ++j) {
        double v = std::abs(T[(size_t)k * n + j]);
        if (v > bestAbs) {
            bestAbs = v;
            best = j;
        }
    }
    return best;
}

// Перестановка столбцов c1 и c2 транспонированной матрицы
void swapTransposed(std::vector<double>& T, int n, int c1, int c2) {
    if (c1 == c2) return;
    for (int j = 0; j < n; ++j) {
        std::swap(T[(size_t)j * n + c1], T[(size_t)j * n + c2]);
    }
}

// Прямой ход метода Гаусса: обновление подматрицы
void eliminate(std::vector<double>& T, int n, int k) {
    double pivot = T[(size_t)k * n + k];
    for (int j = k + 1; j < n; ++j) {
        double factor = T[(size_t)j * n + k] / pivot;
        T[(size_t)j * n + k] = 0.0;
        for (int i = k + 1; i < n; ++i) {
            T[(size_t)j * n + i] -= factor * T[(size_t)k * n + i];
        }
    }
}

// Вычисление определителя через сумму логарифмов
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

    // Транспонируем матрицу (как в оригинале)
    std::vector<double> T(matrixSize);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T[(size_t)j * n + i] = matrix[(size_t)i * n + j];
        }
    }

    // ============ МЕТОД ГАУССА ===============
    int swaps = 0;
    bool singular = false;
    double totalMs = 0;

    for (int k = 0; k < n; ++k) {
        int pivotRow = findPivotTransposed(T, n, k);

        double maxAbs = std::abs(T[(size_t)k * n + pivotRow]);
        if (maxAbs < EPS) {
            singular = true;
            break;
        }

        if (k == n - 1) continue;

        if (pivotRow != k) {
            swapTransposed(T, n, k, pivotRow);
            ++swaps;
        }

        auto start = std::chrono::steady_clock::now();
        eliminate(T, n, k);
        auto stop = std::chrono::steady_clock::now();
        totalMs += std::chrono::duration<double, std::milli>(stop - start).count();
    }

    // ============ РЕЗУЛЬТАТЫ ===============
    double determinant = 0.0;
    if (!singular) {
        std::vector<double> diagonal(n, 0.0);
        for (int i = 0; i < n; ++i) {
            diagonal[i] = T[(size_t)i * n + i];
        }
        determinant = determinantLog(diagonal, swaps);
    }

    std::cout << determinant << '\n';
    std::cout << totalMs << '\n';

    return 0;
}