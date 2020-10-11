#include "core/label_dataset.h"

#include <string>
#include <iostream>

namespace naivebayes {

LabelDataset::LabelDataset() {

}

void LabelDataset::Add(size_t label) {
  labels.push_back(label);
}

size_t LabelDataset::GetLabel(size_t index) {
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

size_t LabelDataset::GetDatasetSize() {
  return labels.size();
}

}
