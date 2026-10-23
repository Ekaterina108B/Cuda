#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cfloat>
#include <chrono>

// ============================================================
// ХЭЛПЕРЫ
// ============================================================

// Хэлпер вывода ошибки
void formatError(const std::string& context, const std::string& err) {
    std::cerr << "ERROR: " << context << ": " << err << '\n';
}

// ============================================================
// CPU
// ============================================================

int main() {
    // ============ НАЧАЛО ===============
    // Получаем данные из стандартного ввода
    std::string inFile, outFile;
    std::cin >> inFile >> outFile;

    // Открываем входной файл на чтение
    std::ifstream fin(inFile, std::ios::binary);
    if (!fin) {
        formatError("FileIn", "can not open");
        return 0;
    }
    
    // Читаем ширину и высоту картинки
    uint32_t W = 0, H = 0;
    fin.read(reinterpret_cast<char*>(&W), 4);
    fin.read(reinterpret_cast<char*>(&H), 4);
    if (W == 0 || H == 0) {
        formatError("Input", "W or H is zero");
        fin.close();
        return 0;
    }
    if (!fin) {
        fin.close();
        formatError("Input", "short header");
        return 0;
    }
    if ((uint64_t)W * H > 400000000ull) {
        formatError("Out of bounds", "W * H > 400000000u");
        return 0;
    }

    // Загружаем все пиксели и закрываем чтение файла
    int pixelsCount = static_cast<int>(W) * H;
    int bytes = static_cast<int>(W) * H * 4;
    std::vector<unsigned char> image(bytes);
    fin.read(reinterpret_cast<char*>(image.data()), bytes);
    if (!fin) {
        formatError("Input", "short read");
        fin.close();
        return 0;
    }
    fin.close();

    // Работа с кластерами
    int nc;
    float avg[32*3] = {};
    std::cin >> nc;
    if ((0 >= nc) || (nc > 32)) {
        formatError("Input", "nc is forbidden");
        fin.close();
        return 0;
    }

    for (int j = 0; j < nc; ++j){
        int np;
        std::cin >> np;
        if ((0 >= np) || (np > 524288)) {
            formatError("Input", "npj is forbidden");
            fin.close();
            return 0;
        }

        double sumR = 0.0;
        double sumG = 0.0;
        double sumB = 0.0;

        for (int i = 0; i < np; ++i) {
            int x, y;
            std::cin >> x >> y;
            if ((x < 0) || (x >= W) || (y < 0) || (y >= H)) {
                formatError("Input", "x or y is forbidden");
                fin.close();
                return 0;
            }
            int idx = (static_cast<size_t>(y) * W + x)*4;
            sumR += image[idx+0];
            sumG += image[idx+1];
            sumB += image[idx+2];
        }
        avg[j*3+0] = static_cast<float>(sumR / np);
        avg[j*3+1] = static_cast<float>(sumG / np);
        avg[j*3+2] = static_cast<float>(sumB / np);
    }

    // ============ ЗАПУСК РАСЧЁТА ===============
    auto start = std::chrono::steady_clock::now();
    for (int p = 0; p < pixelsCount; ++p) {
        int idx = p*4;
        float minD = FLT_MAX; int js = -1;

        for (int j = 0; j < nc; ++j) {
            float d2 = (avg[j*3+0]-image[idx+0])*(avg[j*3+0]-image[idx+0]) + (avg[j*3+1]-image[idx+1])*(avg[j*3+1]-image[idx+1]) + (avg[j*3+2]-image[idx+2])*(avg[j*3+2]-image[idx+2]);
            if (d2 < minD) {
                minD = d2;
                js = j;
            }
        }

        image[idx+3] = static_cast<unsigned char>(js);
    }
    auto stop = std::chrono::steady_clock::now();

    // ============ РЕЗУЛЬТАТЫ ===============
    // Время выполнения
    double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();
    
        std::ofstream fout(outFile, std::ios::binary);
    if (!fout) {
        formatError("FileOut", "can not open");
        return 0;
    }
    fout.write(reinterpret_cast<const char*>(&W), 4);
    fout.write(reinterpret_cast<const char*>(&H), 4);
    fout.write(reinterpret_cast<const char*>(image.data()), bytes);
    fout.close();

    std::cout << "W = " << W << ", H = " << H << '\n';
    std::cout << "Количество кластеров: " << nc << '\n';
    std::cout << "Time [ms]: " << milliseconds << '\n';

    return 0;
}