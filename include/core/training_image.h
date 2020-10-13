#pragma once
#include <cstddef>
#include <vector>

#include "image_dataset.h"

namespace naivebayes {

/**
 * Represents a singular training image and stores data about its pixels and its
 * designated label
 */
struct TrainingImage {
  explicit TrainingImage(const naivebayes::Image &pixels, const size_t &label);

  /**
   * Determines if the pixel at (row, col) is shaded or not
   * @param row The pixel's row
   * @param col The pixel's column
   * @return True if the pixel at (row, col) is shaded, false otherwise
   */
  bool IsShaded(const size_t &row, const size_t &col) const;

  naivebayes::Image pixels_;
  size_t label_;
  size_t image_size_;
};

}  // namespace naivebayes