#include <cstddef>
#include <vector>

namespace naivebayes {

class TrainingImage {
 public:
  explicit TrainingImage(std::vector<std::vector<char>> image_pixels);
  bool IsShaded(size_t x, size_t y);

 private:
  std::vector<std::vector<char>> image_pixels_;
};

}  // namespace naivebayes