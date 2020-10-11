#include <core/training_image.h>

naivebayes::TrainingImage::TrainingImage(Image pixels, size_t label)
    : pixels_(std::move(pixels)), label_(label) {
}

bool naivebayes::TrainingImage::IsShaded(size_t x, size_t y) {
  char pixel = pixels_[x][y];
  return pixel == '+' || pixel == '#';
}

size_t naivebayes::TrainingImage::GetLabel() {
  return label_;
}