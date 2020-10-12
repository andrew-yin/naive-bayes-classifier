#pragma once
#include <cstddef>
#include <vector>

namespace naivebayes {

typedef std::vector<std::vector<char>> Image;

class TrainingImage {
 public:
  explicit TrainingImage(const naivebayes::Image &pixels, const size_t &label);

  bool IsShaded(const size_t &x, const size_t &y) const;
  size_t GetLabel() const;
  size_t GetImageSize() const;

 private:
  naivebayes::Image pixels_;
  size_t label_;
  size_t image_size_;
};

}  // namespace naivebayes