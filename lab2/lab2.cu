#include <iostream>
#include <string>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <vector>

__global__ void ssaa(
    const uchar4* __restrict__ in,
    uchar4* __restrict__ out,
    uint32_t W, uint32_t w, uint32_t h, uint32_t kw, uint32_t kh
) {
    uint64_t n = (uint64_t)kw * kh;

    uint32_t step_x = gridDim.x * blockDim.x;
    uint32_t step_y = gridDim.y * blockDim.y;

    uint32_t j0 = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i0 = blockIdx.y * blockDim.y + threadIdx.y;

    for (uint32_t i = i0; i < h; i += step_y) {
        for (uint32_t j = j0; j < w; j += step_x) {
            uint64_t sr = 0, sg = 0, sb = 0, sa = 0;

            for (uint32_t y = 0; y < kh; ++y) {
                uint32_t sy = i * kh + y;
                for (uint32_t x = 0; x < kw; ++x) {
                    uint32_t sx = j * kw + x;
                    uchar4 p = in[(uint64_t)sy * W + sx];
                    sr += p.x;
                    sg += p.y;
                    sb += p.z;
                    sa += p.w;
                }
            }
            uchar4 res;
            res.x = (unsigned char)(sr / n);
            res.y = (unsigned char)(sg / n);
            res.z = (unsigned char)(sb / n);
            res.w = (unsigned char)(sa / n);
            out[i * w + j] = res;
        }
    }
}

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

// Функция освобождения памяти для gpu
template <typename... Pointers>
void closeGPU(Pointers*... pointers) {
    (cudaFree(pointers), ...);
}

int main() {
    // ============ НАЧАЛО ===============
    // Получаем данные из стандартного ввода
    std::string inFile, outFile;
    uint32_t wn, hn;

    if (!(std::cin >> inFile >> outFile >> wn >> hn)){
        formatError("Input", "input error");
        return 0;
    }
    if ((uint64_t)wn * hn > 400000000ull) {
        formatError("Out of bounds", "wn * hn > 400000000u");
        return 0;
    }
    if (wn == 0 || hn == 0) {
        formatError("Input", "wn or hn is zero");
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
    if (W % wn != 0 || H % hn != 0) {
        formatError("Input", "dimensions not divisible");
        fin.close();
        return 0;
    }

    // Загружаем все пиксели и закрываем чтение файла
    size_t inBytes  = (size_t)W * H * 4;
    size_t outBytes = (size_t)wn * hn * 4;
    std::vector<unsigned char> pixelsIn(inBytes);
    fin.read(reinterpret_cast<char*>(pixelsIn.data()), inBytes);
    if (!fin) {
        formatError("Input", "short read");
        fin.close();
        return 0;
    }
    fin.close();

    // ============ НАСТРОЙКА CUDA ===============
    // Выделение памяти
    unsigned char* d_in; unsigned char* d_out;
    if (!checkCuda(cudaMalloc(&d_in, inBytes), "cudaMalloc d_in")){
        return 0;
    }
    if (!checkCuda(cudaMalloc(&d_out, outBytes), "cudaMalloc d_out")){
        closeGPU(d_in);
        return 0;
    }

    // Копирование памяти
    if (!checkCuda(cudaMemcpy(d_in, pixelsIn.data(), inBytes, cudaMemcpyHostToDevice), "copy pixelIn to GPU")) {
        closeGPU(d_in, d_out);
        return 0;
    }


    // ============ ЗАПУСК CUDA ===============
    uint32_t kw = W/wn; uint32_t kh = H/hn;
    dim3 BLOCK(32, 32);
    dim3 GRID(16, 16);
    ssaa<<<GRID, BLOCK>>>(reinterpret_cast<const uchar4*>(d_in), reinterpret_cast<uchar4*>(d_out), W, wn, hn, kw, kh);

    // Проверяем, успешно ли запустилось ядро
    if (!checkCuda(cudaGetLastError(), "kernel launch")) {
        closeGPU(d_in, d_out);
        return 0;
    }
    // Ждём завершения ядра и проверяем ошибки во время выполнения
    if (!checkCuda(cudaDeviceSynchronize(), "kernel execution")) {
        closeGPU(d_in, d_out);
        return 0;
    }

    // ============ РЕЗУЛЬТАТЫ ===============
    std::vector<unsigned char> pixelsOut(outBytes);
    if (!checkCuda(cudaMemcpy(pixelsOut.data(), d_out, outBytes, cudaMemcpyDeviceToHost), "copy pixelsOut to CPU")) {
        closeGPU(d_in, d_out);
        return 0;
    }

    std::ofstream fout(outFile, std::ios::binary);
    if (!fout) {
        formatError("FileOut", "can not open");
        closeGPU(d_in, d_out);
        return 0;
    }
    fout.write(reinterpret_cast<const char*>(&wn), 4);
    fout.write(reinterpret_cast<const char*>(&hn), 4);
    fout.write(reinterpret_cast<const char*>(pixelsOut.data()), outBytes);
    fout.close();

    closeGPU(d_in, d_out);
    return 0;
}