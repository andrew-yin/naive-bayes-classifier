#pragma once
#include <cstddef>
#include <vector>

#include "image_dataset.h"

namespace naivebayes {

struct TrainingImage {
  explicit TrainingImage(const naivebayes::Image &pixels, const size_t &label);

  bool IsShaded(const size_t &x, const size_t &y) const;

  naivebayes::Image pixels_;
  size_t label_;
  size_t image_size_;
};

}  // namespace naivebayes