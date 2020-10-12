#include <fstream>
#include <vector>

#include "training_image.h"

namespace naivebayes {

class ImageDataset {
 public:
  ImageDataset();

  void Add(const Image &image);
  Image GetImage(const size_t &index) const;
  size_t GetDatasetSize() const;

  friend std::istream &operator>>(std::istream &in, ImageDataset &images);

 private:
  std::vector<Image> images;
};

}  // namespace naivebayes