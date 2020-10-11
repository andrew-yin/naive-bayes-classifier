#pragma once
#include <fstream>
#include <vector>

namespace naivebayes {

class LabelDataset {
 public:
  LabelDataset();

  void Add(size_t label);
  size_t GetLabel(size_t index);
  size_t GetDatasetSize();

  friend std::istream &operator>>(std::istream &in, LabelDataset &labels);

 private:
  std::vector<size_t> labels;
};

}  // namespace naivebayes