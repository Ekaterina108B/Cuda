#include <iostream>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

// ============================================================
// ЯДРО CUDA
// ============================================================

__global__ void ssaa(
    cudaTextureObject_t in,
    uchar4* __restrict__ out,
    uint32_t w, uint32_t h, uint32_t kw, uint32_t kh
) {
    // Количество пикселей в блоке сжатия
    uint64_t n = (uint64_t)kw * kh;

    // Размер шага для потоков
    uint32_t step_x = gridDim.x * blockDim.x;
    uint32_t step_y = gridDim.y * blockDim.y;

    // Координаты начального пикселя для потока
    uint32_t j0 = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i0 = blockIdx.y * blockDim.y + threadIdx.y;

    // Grid-stride loop по строкам
    for (uint32_t i = i0; i < h; i += step_y) {
        // Grid-stride loop по столбцам
        for (uint32_t j = j0; j < w; j += step_x) {

            uint64_t sr = 0, sg = 0, sb = 0, sa = 0;

            // Обход по пикселям в блоке сжатия
            for (uint32_t y = 0; y < kh; ++y) {
                uint32_t sy = i * kh + y;
                for (uint32_t x = 0; x < kw; ++x) {
                    uint32_t sx = j * kw + x;
                    uchar4 p = tex2D<uchar4>(in, (float)sx+0.5f, (float)sy+0.5f);
                    sr += p.x;
                    sg += p.y;
                    sb += p.z;
                    sa += p.w;
                }
            }

            // Фильтрация и запись на выход
            uchar4 res;
            res.x = (unsigned char)(sr / n);
            res.y = (unsigned char)(sg / n);
            res.z = (unsigned char)(sb / n);
            res.w = (unsigned char)(sa / n);
            out[i * w + j] = res;
        }
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
    fin.read(reinterpret_cast<char*>(pixelsIn.data()), inBytes);
    if (!fin) {
        formatError("Input", "short read");
        fin.close();
        return 0;
    }
    fin.close();

    // ============ НАСТРОЙКА CUDA ===============
    uint32_t kw = W/w; uint32_t kh = H/h;

    // Сборка памяти для входной картинки
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>(); // Формат одного элемента -- пиксель

    cudaArray_t d_pixelsIn = nullptr;
    if (!checkCuda(
        cudaMallocArray(&d_pixelsIn, &channelDesc, W, H),
        "cudaMallocArray")) {
            return 0;
    }
    if (!checkCuda(
        cudaMemcpy2DToArray(d_pixelsIn, 0, 0, pixelsIn.data(), (size_t)W * sizeof(uchar4), (size_t)W * sizeof(uchar4), H, cudaMemcpyHostToDevice),
        "copy d_pixelsIn to CUDA array")) {
            cudaFreeArray(d_pixelsIn);
            return 0;
    }

    // Сборка текстуры для входной картинки
    cudaResourceDesc resourceDesc{}; // Откуда читаем данные (правила для каких данных объявляем)
    resourceDesc.resType = cudaResourceTypeArray; // Тип данных для чтения
    resourceDesc.res.array.array = d_pixelsIn; // Сами данные

    cudaTextureDesc textureDesc{}; // Как читаем данные
    textureDesc.addressMode[0] = cudaAddressModeClamp; // Выход за пределы -- брать самый крайний пиксель
    textureDesc.addressMode[1] = cudaAddressModeClamp; // 0 - x, 1 -- y
    textureDesc.filterMode = cudaFilterModePoint; // Выбор ближайшего по координате пикселя (без интерполяции)
    textureDesc.readMode = cudaReadModeElementType; // Возврат формата из ресурса, без приведения к чему-то иному
    textureDesc.normalizedCoords = 0; // Без нормирования координат к [0, 1]

    cudaTextureObject_t tex = 0;
    if (!checkCuda(
        cudaCreateTextureObject(&tex, &resourceDesc, &textureDesc, nullptr),
        "cudaCreateTextureObject")) {
            cudaFreeArray(d_pixelsIn);
            return 0;
    }
    
    // Выделение памяти на выход
    uchar4* d_pixelsOut;
    if (!checkCuda(
        cudaMalloc(&d_pixelsOut, outBytes),
        "cudaMalloc d_pixelsOut")){
            cudaDestroyTextureObject(tex);
            cudaFreeArray(d_pixelsIn);
            return 0;
    }

     // Создание временных точек
    cudaEvent_t start, stop;
    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate start")) {
        cudaDestroyTextureObject(tex);
        cudaFreeArray(d_pixelsIn);
        cudaFree(d_pixelsOut);
        return 0;
    }
    if (!checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop")) {
        cudaEventDestroy(start);
        cudaFree(d_pixelsOut);
        cudaDestroyTextureObject(tex);
        cudaFreeArray(d_pixelsIn);
        return 0;
    }

    // ============ ЗАПУСК ГИПЕРЯДРА ===============
    dim3 BLOCK(32, 32); // Координаты Ока
    dim3 GRID(32, 32); // Координаты Квантовой Луны
    if (!checkCuda(cudaEventRecord(start), "cudaEventRecord start")) { // Временная петля в 22 минуты запущена
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_pixelsOut);
        cudaDestroyTextureObject(tex);
        cudaFreeArray(d_pixelsIn);
        return 0;
    }
    ssaa<<<GRID, BLOCK>>>(tex, d_pixelsOut, w, h, kw, kh); // Корабль номаи отправлен в путь

    // Проверяем, успешно ли запустилось ядро
    if (!checkCuda(cudaGetLastError(), "kernel launch")) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_pixelsOut);
        cudaDestroyTextureObject(tex);
        cudaFreeArray(d_pixelsIn);
        return 0;
    }

    // Временная стоп-точка
    if (!checkCuda(cudaEventRecord(stop), "cudaEventRecord stop")) { // Хранилище номаи обесточено
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_pixelsOut);
        cudaDestroyTextureObject(tex);
        cudaFreeArray(d_pixelsIn);
        return 0;
    }

    // Ждём завершения ядра и проверяем ошибки во время выполнения
    if (!checkCuda(cudaEventSynchronize(stop), "kernel execution")) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_pixelsOut);
        cudaDestroyTextureObject(tex);
        cudaFreeArray(d_pixelsIn);
        return 0;
    }

    // ============ РЕЗУЛЬТАТЫ ===============
    // Время выполнения
    float milliseconds = 0;
    if (!checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapseTime not calculated")) {
        cudaEventDestroy(start); cudaEventDestroy(stop);
        cudaFree(d_pixelsOut);
        cudaDestroyTextureObject(tex);
        cudaFreeArray(d_pixelsIn);
        return 0;
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);
    
    std::vector<unsigned char> pixelsOut(outBytes);
    if (!checkCuda(cudaMemcpy(pixelsOut.data(), d_pixelsOut, outBytes, cudaMemcpyDeviceToHost), "copy pixelsOut to CPU")) {
        cudaFree(d_pixelsOut);
        cudaDestroyTextureObject(tex);
        cudaFreeArray(d_pixelsIn);
        return 0;
    }

    cudaFree(d_pixelsOut);
    cudaDestroyTextureObject(tex);
    cudaFreeArray(d_pixelsIn);

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