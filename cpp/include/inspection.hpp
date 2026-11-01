#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace inspection {
struct Image {
    int width, height;
    std::vector<std::uint8_t> pixels;
};
struct Box { int x, y, width, height, pixels; };
struct Result { std::vector<std::uint8_t> mask; std::vector<Box> boxes; };
Image read_pgm(const std::string& path);
// Eight-connected, handwritten BFS. Area is foreground pixel count, not contour area.
std::vector<Box> components(const std::vector<std::uint8_t>& mask, int width, int height, int min_pixels);
// Replicated-border median 3x3, separable 31x31 sigma=5 Gaussian with REFLECT_101,
// double-precision absolute residual > threshold, no morphology.
Result inspect(const Image& image, double threshold = 8, int min_pixels = 10);
}
