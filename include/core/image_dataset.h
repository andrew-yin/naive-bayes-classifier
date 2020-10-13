#pragma once
#include <fstream>
#include <vector>

namespace naivebayes {

typedef std::vector<std::vector<char>> Image;

/**
 * Represents a dataset of images and overloads input to generate the dataset
 */
struct ImageDataset {
  friend std::istream &operator>>(std::istream &in, ImageDataset &images);
  std::vector<Image> images_;
};

}  // namespace naivebayes