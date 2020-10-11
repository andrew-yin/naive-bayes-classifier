#pragma once
#include <cstddef>
#include <vector>

namespace naivebayes {

typedef std::vector<std::vector<char>> Image;

class TrainingImage {
 public:
  explicit TrainingImage(naivebayes::Image pixels, size_t label);

  bool IsShaded(size_t x, size_t y);
  size_t GetLabel();
  size_t GetImageSize();

 private:
  naivebayes::Image pixels_;
  size_t label_;
  size_t image_size_;
};

}  // namespace naivebayes