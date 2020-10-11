#include <core/training_image.h>

namespace naivebayes {

TrainingImage::TrainingImage(Image pixels, size_t label)
    : pixels_(std::move(pixels)),
      label_(label),
      image_size_(pixels.at(0).size()) {
}

bool TrainingImage::IsShaded(size_t x, size_t y) {
  char pixel = pixels_[x][y];
  return pixel == '+' || pixel == '#';
}

size_t TrainingImage::GetLabel() {
  return label_;
}

size_t TrainingImage::GetImageSize() {
  return image_size_;
}

}  // namespace naivebayes
