#include <core/image_dataset.h>

#include <iostream>
#include <string>

namespace naivebayes {

std::istream &operator>>(std::istream &in, ImageDataset &images) {
  std::string line;
  while (getline(in, line)) {
    size_t image_size = line.size();
    Image image(image_size);

    image[0] = std::vector<char>(line.begin(), line.end());
    for (size_t i = 1; i < image_size; i++) {
      getline(in, line);
      image[i] = std::vector<char>(line.begin(), line.end());
    }

    images.images_.push_back(image);
  }

  return in;
}

}  // namespace naivebayes