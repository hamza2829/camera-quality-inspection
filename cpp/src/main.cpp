#include "inspection.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) throw std::invalid_argument("Usage: inspection_cli image.pgm [threshold=8] [min_pixels=10] [repeat=1]");
        const double threshold = argc > 2 ? std::stod(argv[2]) : 8;
        const int min_pixels = argc > 3 ? std::stoi(argv[3]) : 10;
        const int repeat = argc > 4 ? std::stoi(argv[4]) : 1;
        if (repeat < 1 || repeat > 100) throw std::invalid_argument("Repeat must be 1..100");
        const auto image = inspection::read_pgm(argv[1]);
        inspection::Result result;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat; ++i) result = inspection::inspect(image, threshold, min_pixels);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-start).count()/repeat;
        std::cout << std::setprecision(10) << "{\"width\":" << image.width << ",\"height\":" << image.height
                  << ",\"processing_ms\":" << ms << ",\"mask\":[";
        for (std::size_t i = 0; i < result.mask.size(); ++i) { if (i) std::cout << ','; std::cout << static_cast<int>(result.mask[i]); }
        std::cout << "],\"boxes\":[";
        for (std::size_t i = 0; i < result.boxes.size(); ++i) {
            const auto& box = result.boxes[i];
            if (i) std::cout << ',';
            std::cout << "{\"x\":" << box.x << ",\"y\":" << box.y << ",\"width\":" << box.width
                      << ",\"height\":" << box.height << ",\"pixels\":" << box.pixels << '}';
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
