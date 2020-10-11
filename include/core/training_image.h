#include <cstddef>
#include <vector>

namespace naivebayes {

class TrainingImage {
 public:
  explicit TrainingImage(std::vector<std::vector<char>> image_pixels, size_t label);
  bool IsShaded(size_t x, size_t y);
  size_t GetLabel();

 private:
  std::vector<std::vector<char>> image_pixels_;
  size_t label_;
};

}  // namespace naivebayes