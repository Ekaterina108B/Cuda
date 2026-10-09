#include <iostream>
#include <string>
#include <cstdint>
#include <chrono>
#include <fstream>
#include <vector>

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
    uint32_t w, h;

    if (!(std::cin >> inFile >> outFile >> w >> h)){
        formatError("Input", "input error");
        return 0;
    }
    if ((uint64_t)w * h > 400000000ull) {
        formatError("Out of bounds", "w * h > 400000000u");
        return 0;
    }
    if (w == 0 || h == 0) {
        formatError("Input", "w or h is zero");
        return 0;
    }

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
    if (W % w != 0 || H % h != 0) {
        formatError("Input", "dimensions not divisible");
        fin.close();
        return 0;
    }
    if ((uint64_t)W * H > 400000000ull) {
        formatError("Out of bounds", "W * H > 400000000u");
        return 0;
    }

    // Загружаем все пиксели и закрываем чтение файла
    size_t inBytes  = (size_t)W * H * 4;
    size_t outBytes = (size_t)w * h * 4;
    std::vector<unsigned char> pixelsIn(inBytes);
     std::vector<unsigned char> pixelsOut(outBytes);
    fin.read(reinterpret_cast<char*>(pixelsIn.data()), inBytes);
    if (!fin) {
        formatError("Input", "short read");
        fin.close();
        return 0;
    }
    fin.close();

    // ============ ЗАПУСК РАСЧЁТА ===============
    uint32_t kw = W/w; uint32_t kh = H/h;
    uint64_t n = (uint64_t)kw * kh;

    auto start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < h; ++i) {
        for (uint32_t j = 0; j < w; ++j) {
            uint64_t sr = 0, sg = 0, sb = 0, sa = 0;

            // Обход по пикселям в блоке сжатия
            for (uint32_t y = 0; y < kh; ++y) {
                uint32_t sy = i * kh + y;
                for (uint32_t x = 0; x < kw; ++x) {
                    uint32_t sx = j * kw + x;
                    size_t base = ((size_t)sy * W + sx) * 4;
                    sr += pixelsIn[base + 0];
                    sg += pixelsIn[base + 1];
                    sb += pixelsIn[base + 2];
                    sa += pixelsIn[base + 3];
                }
            }

            // Фильтрация и запись на выход
            size_t outBase = ((size_t)i * w + j) * 4;
            pixelsOut[outBase + 0] = (unsigned char)(sr / n);
            pixelsOut[outBase + 1] = (unsigned char)(sg / n);
            pixelsOut[outBase + 2] = (unsigned char)(sb / n);
            pixelsOut[outBase + 3] = (unsigned char)(sa / n);
        }
    }
    auto stop = std::chrono::steady_clock::now();

    // ============ РЕЗУЛЬТАТЫ ===============
    // Время выполнения
    double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();

    std::cout << W << "x" << H << " превращаем в " << w << "x" << h << '\n';
    std::ofstream fout(outFile, std::ios::binary);
    if (!fout) {
        formatError("FileOut", "can not open");
        return 0;
    }
    fout.write(reinterpret_cast<const char*>(&w), 4);
    fout.write(reinterpret_cast<const char*>(&h), 4);
    fout.write(reinterpret_cast<const char*>(pixelsOut.data()), outBytes);
    fout.close();
    std::cout << "Time [ms]: " << milliseconds << '\n';

    return 0;
}