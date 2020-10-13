#pragma once
#include <fstream>
#include <vector>

namespace naivebayes {

struct LabelDataset {
  friend std::istream &operator>>(std::istream &in, LabelDataset &labels);
  std::vector<size_t> labels_;
};

}  // namespace naivebayes