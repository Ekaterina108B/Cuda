#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << std::scientific << std::setprecision(6);
    
    int n;
    std::cin >> n;

    std::vector<float> arr(n);
    for (int i = 0; i < n; i++) {
        std::cin >> arr[i];
    }

    // Сортировка пузырьком
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - 1 - i; j++) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }

    for (int i = 0; i < n; i++) {
        std::cout << arr[i];
        if (i < n - 1)
            std::cout << " ";
    }
    std::cout << '\n';

    return 0;
}