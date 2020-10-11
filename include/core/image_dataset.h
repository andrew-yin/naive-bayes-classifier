#include <fstream>
#include <vector>

#include "training_image.h"

namespace naivebayes {

class ImageDataset {
 public:
  ImageDataset();

  void Add(Image image);
  Image GetImage(size_t index);
  size_t GetDatasetSize();

  friend std::istream &operator>>(std::istream &in, ImageDataset &images);

 private:
  std::vector<Image> images;
};

}  // namespace naivebayes