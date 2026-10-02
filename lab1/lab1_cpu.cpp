#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>
#include <chrono>

// Функция освобождения памяти для cpu
template <typename... Pointers>
void closeCPU(Pointers*... pointers) {
    (delete[] pointers, ...);
}


int main() {
    // Настройки
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout << std::scientific << std::setprecision(10);
    
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

    // ============ ЗАПУСК РАСЧЁТА ===============
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
        result[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
    auto stop = std::chrono::steady_clock::now();
    double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
    

    // ============ РЕЗУЛЬТАТЫ ===============
    // Вывод
    for (int i = 0; i < n; ++i) {
        std::cout << result[i] << " ";
    }
    std::cout << '\n' << "Time [ms]: " << milliseconds << '\n';

    // Очистка
    closeCPU(a, b, result);

    return 0;
}