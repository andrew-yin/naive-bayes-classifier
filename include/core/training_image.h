#include <cstddef>
#include <vector>

namespace naivebayes {

class TrainingImage {
 public:
  TrainingImage(size_t image_size);
  bool IsShaded(size_t x, size_t y);

 private:
  std::vector<std::vector<char>> image_pixels_;
  size_t label;
};

}