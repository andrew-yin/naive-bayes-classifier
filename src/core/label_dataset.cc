#include "core/label_dataset.h"

#include <iostream>
#include <string>

namespace naivebayes {

std::istream &operator>>(std::istream &in, LabelDataset &labels) {
  std::string line;
  while (getline(in, line)) {
    size_t label = std::stoi(line);
    labels.labels_.push_back(label);
  }

  return in;
}

}  // namespace naivebayes
