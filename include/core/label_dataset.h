#include <vector>
#include <fstream>

namespace naivebayes {

class LabelDataset {
 public:
  friend std::istream &operator>>(std::istream &in, LabelDataset &labels);
  LabelDataset();
  void Add(size_t label);
  size_t GetLabel(size_t index);
  size_t GetDatasetSize();

 private:
  std::vector<size_t> labels;
};

}