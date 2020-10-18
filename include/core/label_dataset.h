#pragma once
#include <fstream>
#include <vector>

namespace naivebayes {

/**
 * Represents a dataset of labels and overloads input to generate the dataset
 */
struct LabelDataset {
  std::vector<size_t> labels_;
};

std::istream &operator>>(std::istream &in, LabelDataset &labels);

}  // namespace naivebayes