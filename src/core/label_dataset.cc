#include "core/label_dataset.h"

#include <iostream>
#include <string>

namespace naivebayes {

LabelDataset::LabelDataset() {
}

void LabelDataset::Add(const size_t &label) {
  labels.push_back(label);
}

size_t LabelDataset::GetLabel(const size_t &index) const {
  return labels.at(index);
}

std::istream &operator>>(std::istream &in, LabelDataset &labels) {
  std::string line;
  while (getline(in, line)) {
    size_t label = std::stoi(line);
    labels.Add(label);
  }

  return in;
}

size_t LabelDataset::GetDatasetSize() const {
  return labels.size();
}

}  // namespace naivebayes
