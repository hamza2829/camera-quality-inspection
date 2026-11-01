#include "inspection.hpp"
#include <gtest/gtest.h>
#include <fstream>

TEST(Components, EmptyMask) {
    EXPECT_TRUE(inspection::components(std::vector<std::uint8_t>(16, 0), 4, 4, 1).empty());
}
TEST(Components, DiagonalIsEightConnected) {
    std::vector<std::uint8_t> mask(16, 0); mask[0] = mask[5] = mask[10] = 255;
    const auto boxes = inspection::components(mask, 4, 4, 3);
    ASSERT_EQ(boxes.size(), 1u);
    EXPECT_EQ(boxes[0].width, 3); EXPECT_EQ(boxes[0].height, 3); EXPECT_EQ(boxes[0].pixels, 3);
}
TEST(Components, SeparateRegionsAndMinimumSize) {
    std::vector<std::uint8_t> mask(36, 0); mask[0] = mask[1] = mask[35] = 255;
    const auto boxes = inspection::components(mask, 6, 6, 2);
    ASSERT_EQ(boxes.size(), 1u); EXPECT_EQ(boxes[0].x, 0); EXPECT_EQ(boxes[0].width, 2);
}
TEST(Preprocessing, ConstantHasNoDefectsIncludingBorders) {
    const auto result = inspection::inspect({12, 12, std::vector<std::uint8_t>(144, 150)});
    EXPECT_TRUE(result.boxes.empty());
    for (const auto pixel : result.mask) EXPECT_EQ(pixel, 0);
}
TEST(Preprocessing, DarkHoleHasRegionAtKnownLocation) {
    inspection::Image image{32, 32, std::vector<std::uint8_t>(1024, 150)};
    for (int y = 12; y < 20; ++y) for (int x = 12; x < 20; ++x) image.pixels[y*32+x] = 10;
    const auto result = inspection::inspect(image);
    ASSERT_FALSE(result.boxes.empty()); EXPECT_LE(result.boxes[0].x, 12); EXPECT_GE(result.boxes[0].width, 8);
}
TEST(Validation, RejectsMalformedInputs) {
    EXPECT_THROW(inspection::inspect({0, 2, {}}), std::invalid_argument);
    EXPECT_THROW(inspection::inspect({2, 2, {1, 2}}), std::invalid_argument);
    EXPECT_THROW(inspection::components({0}, 1, 1, 0), std::invalid_argument);
    EXPECT_THROW(inspection::inspect({2, 2, {0, 0, 0, 0}}, 0), std::invalid_argument);
}
TEST(Pgm, CommentsAndRasterWhitespacePixel) {
    const char* path = "test_raster.pgm";
    { std::ofstream file(path, std::ios::binary); file << "P5\n# comment\n2 2\n255\n"; const char data[] = {32, 10, 0, 127}; file.write(data, 4); }
    const auto image = inspection::read_pgm(path);
    ASSERT_EQ(image.pixels.size(), 4u); EXPECT_EQ(image.pixels[0], 32); EXPECT_EQ(image.pixels[1], 10);
    std::remove(path);
}
TEST(Pgm, RejectsTruncatedRaster) {
    const char* path = "test_truncated.pgm";
    { std::ofstream file(path, std::ios::binary); file << "P5\n2 2\n255\n"; }
    EXPECT_THROW(inspection::read_pgm(path), std::runtime_error); std::remove(path);
}
