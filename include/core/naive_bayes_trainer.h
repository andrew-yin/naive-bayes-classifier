#include <string>

namespace naivebayes {

class NaiveBayesTrainer {
 public:
  friend std::istream &operator>>(std::istream &in, NaiveBayesTrainer &trainer)
};

}  // namespace naivebayes