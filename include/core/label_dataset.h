#pragma once
#include <fstream>
#include <vector>

namespace naivebayes {

class LabelDataset {
 public:
  LabelDataset();

  void Add(const size_t &label);
  size_t GetLabel(const size_t &index) const;
  size_t GetDatasetSize() const;

  friend std::istream &operator>>(std::istream &in, LabelDataset &labels);

 private:
  std::vector<size_t> labels;
};

}  // namespace naivebayes