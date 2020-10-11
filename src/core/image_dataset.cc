#include "core/image_dataset.h"

#include <iostream>
#include <string>

namespace naivebayes {

naivebayes::ImageDataset::ImageDataset() {
}

void ImageDataset::Add(std::vector<std::vector<char>> image) {
  images.push_back(image);
}

std::vector<std::vector<char>> ImageDataset::GetImage(size_t index) {
  return images.at(index);
}

std::istream &operator>>(std::istream &in, ImageDataset &images) {
  std::string line;
  while (getline(in, line)) {
    size_t image_size = line.size();
    std::vector<std::vector<char>> image(image_size);

    image.at(0) = std::vector<char>(line.begin(), line.end());
    for (size_t i = 1; i < image_size; i++) {
      getline(in, line);
      image.at(i) = std::vector<char>(line.begin(), line.end());
    }

    images.Add(image);
  }

  return in;
}

size_t ImageDataset::GetDatasetSize() {
  return images.size();
}

}  // namespace naivebayes
