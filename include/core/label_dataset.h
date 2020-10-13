#pragma once
#include <fstream>
#include <vector>

namespace naivebayes {

/**
 * Represents a dataset of labels and overloads input to generate the dataset
 */
struct LabelDataset {
  friend std::istream &operator>>(std::istream &in, LabelDataset &labels);
  std::vector<size_t> labels_;
};

}  // namespace naivebayes