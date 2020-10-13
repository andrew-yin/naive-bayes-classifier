#include <core/training_image.h>

namespace naivebayes {

TrainingImage::TrainingImage(const Image &pixels, const size_t &label)
    : pixels_(pixels),
      label_(label),
      image_size_(pixels.at(0).size()) {
}

bool TrainingImage::IsShaded(const size_t &row, const size_t &col) const {
  return pixels_[row][col] != ' ';
}

}  // namespace naivebayes
