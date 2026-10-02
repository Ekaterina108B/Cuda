#include <iostream>
#include <cmath>
#include <iomanip>

int main() {
    const float EPS = 1e-6;
    std::cout << std::fixed << std::setprecision(6);
    
    float a, b, c;
    std::cin >> a >> b >> c;

    if (fabs(a) < EPS) {
        // bx + c = 0
        if (fabs(b) < EPS) {
            // c = 0
            if (fabs(c) < EPS) { std::cout << "any"; }
            else { std::cout << "incorrect"; }
        } else {
            // bx + c = 0, b != 0
            std::cout << -c/b;
        }
    } else {
        // ax^2+bx+c = 0
        float d = b*b - 4*a*c;

        if (d < -EPS) {
            // D < 0
            std::cout << "imaginary";
        } else if (fabs(d) < EPS) {
            // D = 0
            std::cout << -b / (2*a);
        } else {
            // D > 0
            float x1 = (-b + sqrt(d)) / (2 * a);
            float x2 = (-b - sqrt(d)) / (2 * a);
            std::cout << x1 << " " << x2;
        }
    }
    std::cout << '\n';

    return 0;
}