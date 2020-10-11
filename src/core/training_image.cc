#include <core/training_image.h>

#include <utility>

naivebayes::TrainingImage::TrainingImage(
    std::vector<std::vector<char>> image_pixels, size_t label)
    : image_pixels_(std::move(image_pixels)) , label_(label){
}

bool naivebayes::TrainingImage::IsShaded(size_t x, size_t y) {
  char pixel = image_pixels_[x][y];
  return pixel == '+' || pixel == '#';
}

size_t naivebayes::TrainingImage::GetLabel() {
  return label_;
}