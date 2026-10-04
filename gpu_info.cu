#include <iostream>
#include <cuda_runtime.h>

int main() {
    int device = 0;
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << '\n';
        return 0;
    }

    int maxBlocksPerSM = 0;
    cudaDeviceGetAttribute(
        &maxBlocksPerSM,
        cudaDevAttrMaxBlocksPerMultiprocessor,
        device
    );

    int maxSharedPerBlockOptin = 0;
    cudaDeviceGetAttribute(
        &maxSharedPerBlockOptin,
        cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device
    );

    int maxSharedPerSM = 0;
    cudaDeviceGetAttribute(
        &maxSharedPerSM,
        cudaDevAttrMaxSharedMemoryPerMultiprocessor,
        device
    );

    int regsPerSM = 0;
    cudaDeviceGetAttribute(
        &regsPerSM,
        cudaDevAttrMaxRegistersPerMultiprocessor,
        device
    );

    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Compute capability: "
              << prop.major << "." << prop.minor << '\n';
    std::cout << "Графическая память: "
              << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Разделяемая память (по умолчанию): "
              << prop.sharedMemPerBlock / 1024 << " KB\n";
    std::cout << "Разделяемая память (opt-in, на блок): "
              << maxSharedPerBlockOptin / 1024 << " KB\n";
    std::cout << "Разделяемая память (на мультипроцессор): "
              << maxSharedPerSM / 1024 << " KB\n";
    std::cout << "Константная память: "
              << prop.totalConstMem / 1024 << " KB\n";
    std::cout << "Кол-во регистров на блок: "
              << prop.regsPerBlock << '\n';
    std::cout << "Кол-во регистров на мультипроцессор: "
              << regsPerSM << '\n';
    std::cout << "Макс. кол-во потоков на блок: "
              << prop.maxThreadsPerBlock << '\n';
    std::cout << "Макс. размерность блока: "
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << '\n';
    std::cout << "Макс. кол-во потоков на мультипроцессор: "
              << prop.maxThreadsPerMultiProcessor << '\n';
    std::cout << "Макс. кол-во блоков на мультипроцессор: "
              << maxBlocksPerSM << '\n';
    std::cout << "Максимальный размер сетки: "
              << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x "
              << prop.maxGridSize[2] << '\n';
    std::cout << "Мультипроцессоры: "
              << prop.multiProcessorCount << '\n';

    return 0;
}