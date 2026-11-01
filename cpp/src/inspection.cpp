#include "inspection.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace inspection {
namespace {
void validate(int width, int height, std::size_t size) {
    if (width < 1 || height < 1 || width > 4096 || height > 4096 ||
        size != static_cast<std::size_t>(width) * height)
        throw std::invalid_argument("Invalid image dimensions or buffer length");
}
int reflect(int value, int size) {
    if (size == 1) return 0;
    while (value < 0 || value >= size) {
        if (value < 0) value = -value;
        else value = 2 * size - value - 2;
    }
    return value;
}
std::string token(std::istream& input) {
    std::string value;
    while (input >> value) {
        if (value[0] != '#') return value;
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    throw std::runtime_error("Truncated PGM header");
}
}

Image read_pgm(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input || token(input) != "P5") throw std::runtime_error("Expected binary P5 PGM");
    const int width = std::stoi(token(input)), height = std::stoi(token(input));
    if (token(input) != "255") throw std::runtime_error("Only 8-bit maxval=255 PGM is supported");
    validate(width, height, static_cast<std::size_t>(width) * std::max(height, 0));
    const int separator = input.get();
    if (separator == '\r' && input.peek() == '\n') input.get();
    else if (separator != ' ' && separator != '\n' && separator != '\t' && separator != '\r')
        throw std::runtime_error("Missing PGM raster separator");
    Image image{width, height, std::vector<std::uint8_t>(static_cast<std::size_t>(width) * height)};
    input.read(reinterpret_cast<char*>(image.pixels.data()), static_cast<std::streamsize>(image.pixels.size()));
    if (input.gcount() != static_cast<std::streamsize>(image.pixels.size())) throw std::runtime_error("Truncated PGM raster");
    return image;
}

std::vector<Box> components(const std::vector<std::uint8_t>& mask, int width, int height, int min_pixels) {
    validate(width, height, mask.size());
    if (min_pixels < 1) throw std::invalid_argument("min_pixels must be positive");
    std::vector<std::uint8_t> seen(mask.size(), 0);
    std::vector<Box> boxes;
    std::vector<int> queue;
    for (int start = 0; start < static_cast<int>(mask.size()); ++start) {
        if (!mask[start] || seen[start]) continue;
        queue.clear(); queue.push_back(start); seen[start] = 1;
        int xmin = width, xmax = 0, ymin = height, ymax = 0;
        for (std::size_t head = 0; head < queue.size(); ++head) {
            const int x = queue[head] % width, y = queue[head] / width;
            xmin = std::min(xmin, x); xmax = std::max(xmax, x);
            ymin = std::min(ymin, y); ymax = std::max(ymax, y);
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                const int nx = x + dx, ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                const int next = ny * width + nx;
                if (mask[next] && !seen[next]) { seen[next] = 1; queue.push_back(next); }
            }
        }
        if (static_cast<int>(queue.size()) >= min_pixels)
            boxes.push_back({xmin, ymin, xmax-xmin+1, ymax-ymin+1, static_cast<int>(queue.size())});
    }
    std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) {
        return a.x != b.x ? a.x < b.x : a.y < b.y;
    });
    return boxes;
}

Result inspect(const Image& image, double threshold, int min_pixels) {
    validate(image.width, image.height, image.pixels.size());
    if (!std::isfinite(threshold) || threshold <= 0 || threshold >= 255)
        throw std::invalid_argument("Threshold must be finite and inside (0,255)");
    const int w = image.width, h = image.height;
    std::vector<double> smooth(image.pixels.size()), horizontal(smooth.size()), background(smooth.size());
    std::array<int, 9> neighborhood{};
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        int index = 0;
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx)
            neighborhood[index++] = image.pixels[std::clamp(y+dy, 0, h-1)*w + std::clamp(x+dx, 0, w-1)];
        std::nth_element(neighborhood.begin(), neighborhood.begin()+4, neighborhood.end());
        smooth[y*w+x] = neighborhood[4];
    }
    std::array<double, 31> kernel{};
    double sum = 0;
    for (int k = -15; k <= 15; ++k) { kernel[k+15] = std::exp(-k*k/50.0); sum += kernel[k+15]; }
    for (auto& value : kernel) value /= sum;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        for (int k = -15; k <= 15; ++k) horizontal[y*w+x] += kernel[k+15]*smooth[y*w+reflect(x+k, w)];
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x)
        for (int k = -15; k <= 15; ++k) background[y*w+x] += kernel[k+15]*horizontal[reflect(y+k, h)*w+x];
    Result result;
    result.mask.resize(smooth.size());
    for (std::size_t i = 0; i < smooth.size(); ++i)
        result.mask[i] = std::abs(smooth[i]-background[i]) > threshold ? 255 : 0;
    result.boxes = components(result.mask, w, h, min_pixels);
    return result;
}
}
