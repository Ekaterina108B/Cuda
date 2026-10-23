#include <iostream>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cfloat>

// ============================================================
// ЯДРО CUDA
// ============================================================
__constant__ float3 d_avg[32];

__global__ void midimet(
    uchar4* __restrict__ pixels,
    int n, int nc
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    while (idx < n) {
        uchar4 p = pixels[idx];
        float minD = FLT_MAX; int js = -1;

        for (int j = 0; j < nc; ++j) {
            float d2 = (d_avg[j].x-p.x)*(d_avg[j].x-p.x) + (d_avg[j].y-p.y)*(d_avg[j].y-p.y) + (d_avg[j].z-p.z)*(d_avg[j].z-p.z);
            if (d2 < minD) {
                minD = d2;
                js = j;
            }
        }

        pixels[idx].w = static_cast<unsigned char>(js);
        idx += totalThreads;
    }
}

// ============================================================
// ХЭЛПЕРЫ
// ============================================================

// Хэлпер вывода ошибки
void formatError(const std::string& context, const std::string& err) {
    std::cerr << "ERROR: " << context << ": " << err << '\n';
}

// Хэлпер вывода ошибки от cuda
bool checkCuda(cudaError_t error, const std::string& operation) {
    if (error != cudaSuccess) {
        formatError("CUDA error during " + operation, cudaGetErrorString(error));
        return false; 
    }
    return true;
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
    int pixelCount = static_cast<int>(W) * H;
    std::vector<uchar4> image(pixelCount);
    fin.read(reinterpret_cast<char*>(image.data()),
        static_cast<std::streamsize>(pixelCount * sizeof(uchar4))
    );

    if (!fin) {
        formatError("Input", "short read");
        fin.close();
        return 0;
    }
    fin.close();

    // Работа с кластерами
    int nc;
    float3 avg[32] = {};
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
            uchar4 p = image[static_cast<size_t>(y) * W + x];

            sumR += p.x;
            sumG += p.y;
            sumB += p.z;
        }
        avg[j] = make_float3(
            static_cast<float>(sumR / np),
            static_cast<float>(sumG / np),
            static_cast<float>(sumB / np)
        );
    }

    // ============ НАСТРОЙКА CUDA ===============
    // Выделение памяти на выход
    uchar4* d_pixels;
    if (!checkCuda(
        cudaMalloc(&d_pixels, pixelCount * sizeof(uchar4)),
        "cudaMalloc d_pixelsIn")){
            return 0;
    }
    if (!checkCuda(
        cudaMemcpy(d_pixels, image.data(), pixelCount * sizeof(uchar4), cudaMemcpyHostToDevice),
        "copy a to GPU")) {
            cudaFree(d_pixels);
            return 0;
    }

    if (!checkCuda(
        cudaMemcpyToSymbol(d_avg, avg, nc * sizeof(float3)),
        "copy avg to GPU constant")) {
            cudaFree(d_pixels);
            return 0;
    }

    // ============ ЗАПУСК ГИПЕРЯДРА ===============
    dim3 BLOCK(32); // Координаты Ока
    dim3 GRID(16); // Координаты Квантовой Луны
    midimet<<<GRID, BLOCK>>>(d_pixels, pixelCount, nc); // Корабль номаи отправлен в путь

    // Проверяем, успешно ли запустилось ядро
    if (!checkCuda(cudaGetLastError(), "kernel launch")) {
        cudaFree(d_pixels);
        return 0;
    }
    // Ждём завершения ядра и проверяем ошибки во время выполнения
    if (!checkCuda(cudaDeviceSynchronize(), "kernel execution")) {
        cudaFree(d_pixels);
        return 0;
    }

    // ============ РЕЗУЛЬТАТЫ ===============
    if (!checkCuda(
        cudaMemcpy(image.data(), d_pixels, pixelCount * sizeof(uchar4), cudaMemcpyDeviceToHost),
        "copy pixels to CPU")) {
            cudaFree(d_pixels);
            return 0;
    }

    std::ofstream fout(outFile, std::ios::binary);
    if (!fout) {
        formatError("FileOut", "can not open");
        cudaFree(d_pixels);
        return 0;
    }
    fout.write(reinterpret_cast<const char*>(&W), 4);
    fout.write(reinterpret_cast<const char*>(&H), 4);
    fout.write(reinterpret_cast<const char*>(image.data()), pixelCount * sizeof(uchar4));
    fout.close();
    cudaFree(d_pixels);

    return 0;
}