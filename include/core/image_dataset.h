#include <vector>
#include <fstream>

namespace naivebayes {

class ImageDataset {
 public:
  friend std::istream &operator>>(std::istream &in, ImageDataset &images);
  ImageDataset();
  void Add(std::vector<std::vector<char>> image);
  std::vector<std::vector<char>> GetImage(size_t index);
  size_t GetDatasetSize();

 private:
  std::vector<std::vector<std::vector<char>>> images;
};

} //namespace naivebayes