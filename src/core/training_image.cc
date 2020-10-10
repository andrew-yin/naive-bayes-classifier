#include <core/training_image.h>

#include <utility>

naivebayes::TrainingImage::TrainingImage(std::vector<std::vector<char>> image_pixels)
    : image_pixels_(std::move(image_pixels)) {
}

bool naivebayes::TrainingImage::IsShaded(size_t x, size_t y) {
  char pixel = image_pixels_[x][y];
  return pixel == '+' || pixel == '#';
}