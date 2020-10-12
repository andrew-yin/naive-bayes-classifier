#include <core/training_image.h>

namespace naivebayes {

TrainingImage::TrainingImage(const Image &pixels, const size_t &label)
    : pixels_(std::move(pixels)),
      label_(label),
      image_size_(pixels.at(0).size()) {
}

bool TrainingImage::IsShaded(const size_t &x, const size_t &y) const {
  char pixel = pixels_[x][y];
  return pixel == '+' || pixel == '#';
}

size_t TrainingImage::GetLabel() const {
  return label_;
}

size_t TrainingImage::GetImageSize() const {
  return image_size_;
}

}  // namespace naivebayes
